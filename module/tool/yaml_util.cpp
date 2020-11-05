//
// Created by liulei on 2020/5/20.
//

#include "yaml_util.h"

namespace rtf {
    namespace YAMLUtil {

        YAML::Node loadYAML(string yamlPath) {
            YAML::Node node;
            try {
                node = YAML::LoadFile(yamlPath);
            } catch (YAML::BadFile& ex) {
                LOG(ERROR) << "Cannot read file: " << yamlPath;
                throw invalid_argument("yaml file cannot be found. the path is " + yamlPath);
            }
            return node;
        }


        void saveYAML(string yamlPath, YAML::Node node) {
            ofstream out(yamlPath);
            out << node;
        }


        YAML::Node mergeNodes(initializer_list<YAML::Node> nodes) {
            YAML::Node finalNode;
            // foreach the nodes
            for(auto &node : nodes) {
                YAML::NodeType::value nodeType = node.Type();
                for(auto &item: node) {
                    if (nodeType == YAML::NodeType::Map) { // exception
                        finalNode[item.first.as<string>()] = item.second;
                    } else if (nodeType == YAML::NodeType::Sequence) {
                        finalNode.push_back(item.second);
                    }
                }
            }
            return finalNode;
        }


        YAML::Node matrixSerialize(MatrixX matrix) {
            YAML::Node node;

            for(int i=0; i<matrix.rows(); i++) {
                for(int j=0; j<matrix.cols(); j++) {
                    node.push_back(matrix(i, j));
                }
            }

            return node;
        }
    }
}