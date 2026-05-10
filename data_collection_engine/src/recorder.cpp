#include "recorder.hpp"

#include <rmw/rmw.h>

#include <rclcpp/serialized_message.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <rosbag2_storage/storage_options.hpp>

DataRecorder::DataRecorder(rclcpp::Node::SharedPtr parent_node,
                           YAML::Node config)
    : node_(parent_node) {
  if (!config["target_root_dir"]) {
    throw std::runtime_error(
        "DataRecorder: config missing required key 'target_root_dir'");
  }
  target_root_dir_ = config["target_root_dir"].as<std::string>();

  if (!config["topics"] || !config["topics"].IsSequence()) {
    throw std::runtime_error(
        "DataRecorder: config missing required sequence 'topics'");
  }
  for (const auto& entry : config["topics"]) {
    const auto& topic = entry["topic"];
    if (!topic || !topic["name"] || !topic["type"]) {
      throw std::runtime_error(
          "DataRecorder: every topics[] entry must have a 'topic:' sub-map "
          "with 'name' and 'type'");
    }
    topics_.push_back(TopicInfo{
        topic["name"].as<std::string>(),
        topic["type"].as<std::string>(),
        topic["latched"] ? topic["latched"].as<bool>() : false,
    });
  }

  RCLCPP_INFO(node_->get_logger(),
              "DataRecorder configured with %zu topic(s), output dir: %s",
              topics_.size(), target_root_dir_.c_str());
}

void DataRecorder::StartRecording(const std::string& recording_name) {
  if (recording_) {
    RCLCPP_WARN(node_->get_logger(),
                "StartRecording called, but already recording. Stop existing "
                "recording before starting a new one.");
    return;
  }
  writer_ = std::make_unique<rosbag2_cpp::Writer>();

  rosbag2_cpp::ConverterOptions converter_options;
  converter_options.input_serialization_format = rmw_get_serialization_format();
  converter_options.output_serialization_format =
      rmw_get_serialization_format();

  storage_options_.uri = target_root_dir_ + "/" + recording_name;
  storage_options_.storage_id = "mcap";

  writer_->open(storage_options_, converter_options);

  recording_ = true;

  for (const TopicInfo& topic_info : topics_) {
    SetupTopic(topic_info.name, topic_info.type, topic_info.latched);
  }

  RCLCPP_INFO(node_->get_logger(), "Started recording. Opened rosbag at: %s",
              storage_options_.uri.c_str());
}

void DataRecorder::StopRecording() {
  if (writer_ == nullptr) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(writer_mutex_);

    recording_ = false;

    writer_->close();
    writer_.reset();
  }

  subscriptions_.clear();
  RCLCPP_INFO(node_->get_logger(), "Recording finished. Rosbag written to: %s",
              storage_options_.uri.c_str());
}

void DataRecorder::SetupTopic(const std::string& topic_name,
                              const std::string& topic_type, bool latched) {
  rosbag2_storage::TopicMetadata tm;
  tm.name = topic_name;
  tm.type = topic_type;
  tm.serialization_format = rmw_get_serialization_format();

  writer_->create_topic(tm);

  // Long queue to make sure we don't lose any message.
  rclcpp::QoS qos_profile(1000);
  if (latched) {
    qos_profile.transient_local();
  }
  qos_profile.reliable();

  auto sub = node_->create_generic_subscription(
      topic_name, topic_type, qos_profile,
      [this, topic_name, topic_type](
          std::shared_ptr<rclcpp::SerializedMessage> msg,
          const rclcpp::MessageInfo& msg_info) {
        TopicCallback(topic_name, topic_type, msg, msg_info);
      });

  RCLCPP_DEBUG(node_->get_logger(), "Setup recording on topic: %s",
               topic_name.c_str());

  subscriptions_.push_back(sub);
}

void DataRecorder::TopicCallback(const std::string& topic_name,
                                 const std::string& topic_type,
                                 std::shared_ptr<rclcpp::SerializedMessage> msg,
                                 const rclcpp::MessageInfo& msg_info) {
  std::lock_guard<std::mutex> lock(writer_mutex_);
  if (!recording_) {
    return;
  }
  const auto& rmw_info = msg_info.get_rmw_message_info();
  writer_->write(msg, topic_name, topic_type, rmw_info.received_timestamp,
                 rmw_info.source_timestamp);
}
