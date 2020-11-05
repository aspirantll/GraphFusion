//
// Created by liulei on 2020/5/19.
//

#ifndef RTF_FILE_UTIL_H
#define RTF_FILE_UTIL_H

#include <string>
#include <initializer_list>

using namespace std;

namespace rtf {
    namespace FileUtil {

        /**
         * joint paths by file separator
         */
        string joinPath(initializer_list<string> paths);

        /*
         * if exist, then return true
         */
        bool exist(string path);

        /*
         * if not exist, then create
         */
        int createDirectory(string path);
    }
}



#endif //RTF_FILE_UTIL_H
