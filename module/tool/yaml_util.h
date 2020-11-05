//
// Created by liulei on 2020/5/20.
//

#ifndef RTF_YAML_UTIL_H
#define RTF_YAML_UTIL_H

#include <yaml-cpp/yaml.h>
#include <string>
#include <typeinfo>
#include <fstream>
#include <initializer_list>
#include <glog/logging.h>
#include "../datastructure/base_types.h"

using namespace std;

namespace rtf {
    namespace YAMLUtil {
        /**
         * load yaml from file
         * @param yamlPath
         * @return
         */
        YAML::Node loadYAML(string yamlPath);

        /**
         * save yaml node to file
         * @param yamlPath
         * @param node
         */
        void saveYAML(string yamlPath, YAML::Node node);

        /**
         * merge many to one
         * @param one
         * @param another
         * @return
         */
        YAML::Node mergeNodes(initializer_list<YAML::Node> nodes);

        template <class E>
        YAML::Node baseVectorSerialize(vector<E>& elements) {
            YAML::Node node;
            for(E e: elements) {
                node.push_back(e);
            }
            return node;
        }

        template <class E>
        void baseVectorDeserialize(YAML::Node node, vector<E>& elements) {
            if(node.IsNull()) return;
            LOG_ASSERT(node.IsSequence()) << " the node is not sequence!";
            for(int i=0; i<node.size(); i++) {
                elements.emplace_back(node[i].as<E>());
            }
        }

        template <class E>
        YAML::Node baseArraySerialize(E* elements, int len) {
            YAML::Node node;
            for(int i=0; i<len; i++) {
                node.push_back(elements[i]);
            }
            return node;
        }

        template <class E>
        void baseArrayDeserialize(YAML::Node node, E* elements) {
            if(node.IsNull()) return;
            LOG_ASSERT(node.IsSequence()) << " the node is not sequence!";
            for(int i=0; i<node.size(); i++) {
                elements[i] = node[i].as<E>();
            }
        }

        template <class E>
        YAML::Node vectorSerialize(vector<E>& elements) {
            YAML::Node node;
            for(E e: elements) {
                node.push_back(e.serialize());
            }
            return node;
        }

        template <class E>
        void vectorDeserialize(YAML::Node node, vector<E>& elements) {
            if(node.IsNull()) return;
            LOG_ASSERT(node.IsSequence()) << " the node is not sequence!";
            for(int i=0; i<node.size(); i++) {
                elements.emplace_back(node[i]);
            }
        }


        template <class E>
        YAML::Node vectorSerialize(vector<shared_ptr<E>>& elements) {
            YAML::Node node;
            for(shared_ptr<E> e: elements) {
                node.push_back(e->serialize());
            }
            return node;
        }

        template <class E>
        void vectorDeserialize(YAML::Node node, vector<shared_ptr<E>>& elements) {
            if(node.IsNull()) return;
            LOG_ASSERT(node.IsSequence()) << " the node is not sequence!";
            for(int i=0; i<node.size(); i++) {
                elements.emplace_back(allocate_shared<E>(Eigen::aligned_allocator<E>(), node[i]));
            }
        }

        template <class E>
        YAML::Node vectorSerialize(vector<E, Eigen::aligned_allocator<E>>& elements) {
            YAML::Node node;
            for(E e: elements) {
                node.push_back(e.serialize());
            }
            return node;
        }

        template <class E>
        void vectorDeserialize(YAML::Node node, vector<E, Eigen::aligned_allocator<E>>& elements) {
            if(node.IsNull()) return;
            LOG_ASSERT(node.IsSequence()) << " the node is not sequence!";
            for(int i=0; i<node.size(); i++) {
                elements.emplace_back(node[i]);
            }
        }

        template <class E>
        YAML::Node eigenUpperTriangularMatrixSerialize(EigenUpperTriangularMatrix<E>& matrix) {
            YAML::Node node;
            node["matVec"] = vectorSerialize(matrix.matVec);
            node["n"] = matrix.n;
            node["defaultValue"] = matrix.defaultValue.serialize();

            return node;
        }

        template <class E>
        void eigenUpperTriangularMatrixDeserialize(YAML::Node node, EigenUpperTriangularMatrix<E>& matrix) {
            YAMLUtil::vectorDeserialize(node["matVec"], matrix.matVec);
            matrix.n = node["n"].as<int>();
            matrix.defaultValue = E(node["defaultValue"]);
        }


        YAML::Node matrixSerialize(MatrixX matrix);


        template<class M>
        void matrixDeserialize(YAML::Node node, M& m) {
            if(node.IsNull()) return;
            LOG_ASSERT(node.IsSequence()) << " the node is not sequence!";

            for (int i = 0; i < node.size(); i++) {
                m(i / m.rows(), i % m.rows()) = node[i].as<double>();
            }
        }
    }
}


#endif //RTF_YAML_UTIL_H
