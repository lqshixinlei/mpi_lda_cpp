#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

void StrictSplit(const std::string &str, const std::string &dem, std::vector<std::string> *fields) {
    uint32_t pre_pos = 0;
    std::string::size_type pos = 0;
    if (dem.size() == 1) {
        for (uint32_t i = 0;i < str.size();i++) {
            if (str[i] == dem[0]) {
                fields->push_back(str.substr(pre_pos, i-pre_pos));
                pre_pos = i+1;
            }
        }
        if (pre_pos < str.size()) fields->push_back(str.substr(pre_pos));
    }
    else if (dem.size() > 1) {
        while (1) {
            pos = str.find(dem, pre_pos);
            if (pos == str.npos) {
                fields->push_back(str.substr(pre_pos));
                break;
            }
            else if (pos == 0) {
                pre_pos += dem.size();
            }
            else if ((pos + dem.size()) == str.size()) {
                fields->push_back(str.substr(pre_pos, pos-pre_pos));
                break;
            }
            else {
                fields->push_back(str.substr(pre_pos, pos-pre_pos));
                pre_pos = pos + dem.size();
            }
        }
                
    }
}
