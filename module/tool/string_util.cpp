#include "string_util.h"


namespace rtf {
    namespace StringUtil {

        bool hasText(string str) {
            for (char ch : str) {
                if (ch != 0 && ch != ' ' && ch != '\t') {
                    return true;
                }
            }
            return false;
        }


        vector<string> split(string src, char token) {
            vector<string> res;

            string buf = "";
            for (char ch : src) {
                if (ch == token || ch == ' ') {
                    if (buf.size() > 0) {
                        res.push_back(buf);
                        buf = "";
                    }
                } else {
                    buf += ch;
                }
            }
            if (buf.size() > 0) {
                res.push_back(buf);
            }
            return res;
        }

        string join(string token, initializer_list<string> strList) {
            ostringstream out;

            auto strP=strList.begin();
            out << *strP;
            for(++strP; strP!=strList.end(); ++strP) {
                out << token << *strP;
            }
            return out.str();
        }

        vector<int> toIntVec(string src) {
            vector<string> parts = split(src, ',');
            vector<int> intVec;
            for(string part: parts) {
                intVec.push_back(toInt(part));
            }
            return intVec;
        }

        int toInt(string src) {
            istringstream in(src);
            int tar;
            in >> tar;
            return tar;
        }

        float toFloat(string src) {
            istringstream in(src);
            float tar;
            in >> tar;
            return tar;
        }

        double toDouble(string src) {
            istringstream in(src);
            double tar;
            in >> tar;
            return tar;
        }
    }


}
