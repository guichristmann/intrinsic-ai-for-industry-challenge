#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <string>
#include <vector>

class DataRecorder {
 public:
  DataRecorder(rclcpp::Node::SharedPtr parent_node, YAML::Node config);

  void StartRecording(const std::string& recording_name);
  void StopRecording();

 private:
  struct TopicInfo {
    std::string name;
    std::string type;
    bool latched{false};
  };

  void SetupTopic(const std::string& topic_name, const std::string& topic_type,
                  bool latched);
  void TopicCallback(const std::string& topic_name,
                     const std::string& topic_type,
                     std::shared_ptr<rclcpp::SerializedMessage> msg,
                     const rclcpp::MessageInfo& msg_info);

  rclcpp::Node::SharedPtr node_;
  std::string target_root_dir_;
  std::vector<TopicInfo> topics_;

  std::atomic<bool> recording_{false};
  std::mutex writer_mutex_;
  std::unique_ptr<rosbag2_cpp::Writer> writer_;
  rosbag2_storage::StorageOptions storage_options_;
  std::vector<std::shared_ptr<rclcpp::GenericSubscription>> subscriptions_;
};
