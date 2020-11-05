#include <string>
#include <vector>
#include <sstream>
#include <initializer_list>
#include <Eigen/Core>

using namespace std;
namespace rtf {
    namespace StringUtil {

        /**
         * if str contains non-empty character, return true. otherwise, return false.
         * @param str
         * @return
         */
        bool hasText(string str);

        /**
         * split string by token
         * @param src
         * @param token
         * @return
         */
        vector<string> split(string src, char token);

        /**
         * join strings by token
         * @param token
         * @param strList
         * @return
         */
        string join(string token, initializer_list<string> strList);


        /**
         * format the str
         * @param formatStr
         * @return
         */
        // todo format for string class
//        string format(string formatStr);


        /**
         * convert string to int, such as -1,1,2
         * @param src
         * @return
         */
        vector<int> toIntVec(string src);


        /**
         * convert string to int
         * @param src
         * @return
         */
        int toInt(string src);

        /**
         * convert string to float
         * @param src
         * @return
         */
        float toFloat(string src);

        /**
         * convert string to double
         * @param src
         * @return
         */
        double toDouble(string src);

    }
}