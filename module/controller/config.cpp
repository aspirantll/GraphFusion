//
// Created by liulei on 2020/6/5.
//

#include "config.h"
#include "../tool/string_util.h"

namespace rtf {


    shared_ptr<BaseConfig> BaseConfig::config = nullptr;

    BaseConfig::BaseConfig(string workspace, CameraModelType cameraModelType) : workspace(workspace),
                                                                                cameraModelType(cameraModelType) {

    }


    void BaseConfig::initInstance(string workspace, CameraModelType cameraModelType) {
        if (config == nullptr) {
            config = make_shared<BaseConfig>(BaseConfig(workspace, cameraModelType));
        }
    }


    shared_ptr<BaseConfig> BaseConfig::getInstance() {
        if (config == nullptr) {
            throw runtime_error("config is uninitialized. please invoke BaseConfig::initInstance before this method");
        }
        return config;
    }

    GlobalConfig::GlobalConfig(const string &workspace) : workspace(workspace) {
        BaseConfig::initInstance(workspace);
        vocTxtPath = workspace + "/voc.txt";
    }

    void GlobalConfig::loadFromFile(const string &file) {
        YAML::Node node = YAMLUtil::loadYAML(file);
        /*workspace = node["workspace"].as<string>();
        BaseConfig::initInstance(workspace);*/
        kMinMatches = node["kMinMatches"].as<int>();
        kMinInliers = node["kMinInliers"].as<int>();
        rmsThreshold = node["rmsThreshold"].as<float>();
        irThreshold = node["irThreshold"].as<float>();
        maxPnPResidual = node["maxPnPResidual"].as<float>();
        maxAvgCost = node["maxAvgCost"].as<float>();
        costThreshold = node["costThreshold"].as<float>();
        overlapNum = node["overlapNum"].as<int>();
        minInlierRatio = node["minInlierRatio"].as<float>();
        vocTxtPath = node["vocTxtPath"].as<string>();
        upperBoundResidual = node["upperBoundResidual"].as<float>();
    }

    void GlobalConfig::saveToFile(const string &file) {
        YAML::Node node;
        node["workspace"] = workspace;
        node["kMinMatches"] = kMinMatches;
        node["kMinInliers"] = kMinInliers;
        node["rmsThreshold"] = rmsThreshold;
        node["irThreshold"] = irThreshold;
        node["maxPnPResidual"] = maxPnPResidual;
        node["maxAvgCost"] = maxAvgCost;
        node["costThreshold"] = costThreshold;
        node["overlapNum"] = overlapNum;
        node["minInlierRatio"] = minInlierRatio;
        node["vocTxtPath"] = vocTxtPath;
        node["upperBoundResidual"] = upperBoundResidual;

        YAMLUtil::saveYAML(file, node);
    }
}