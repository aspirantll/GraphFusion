//
// Created by liulei on 2020/5/19.
//

#include "file_util.h"
#include "string_util.h"

#ifdef _WIN32
#include <direct.h>
#include <io.h>
const string KFileSeparator="\\";
#elif __linux
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
const string KFileSeparator="/";
#endif

namespace rtf {
    namespace FileUtil {

        string joinPath(initializer_list<string> paths) {
            return StringUtil::join(KFileSeparator, paths);
        }

        bool exist(string path) {
            return access(path.c_str(), 0) != -1;
        }

        int createDirectory(string path) {
            int len = path.length();
            char tmpDirPath[256] = { 0 };
            for (int i = 0; i < len; i++) {
                tmpDirPath[i] = path[i];
                if (tmpDirPath[i] == '\\' || tmpDirPath[i] == '/' || i == len-1)
                {
                    if (!exist(tmpDirPath))
                    {
                        #ifdef _WIN32
                        int ret = mkdir(tmpDirPath);
                        #elif __linux
                        int ret = mkdir(tmpDirPath,  S_IRWXU);
                        #endif
                        if (ret == -1) return ret;
                    }
                }
            }
            return 0;
        }
    }
}