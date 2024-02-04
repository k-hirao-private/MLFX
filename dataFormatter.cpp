#include <fstream>
#include <iostream>
#include <string>
#include "json.hpp"
#include <vector>
#include <map>
using string = std::string;
using json = nlohmann::json;

#define getParamNum() intervals.size() * columns.size() * approximate_dim

string symbol = "USD-JPY";
std::vector<string> intervals{"1day"};
std::vector<string> columns{
    "open",
    "low",
    "high",
    "close",
    "SMA",
    "bollinger_upper",
    "bollinger_lower",
};
int approximate_dim = 4;
string base_interval = "1day";

class ChartRangeError
{
private:
    const char *msg; // 例外を説明するメッセージ
public:
    ChartRangeError(const char *msg) : msg(msg) {} // コンストラクタ
    const char what() { return msg; }              // メッセージを返す
};
std::vector<float> getParams()
{
}
int getLabel()
{
}
void makeInputData(std::map<string, json> data, int start_i, int end_i)
{
    std::vector<std::vector<float>> params;
    params.reserve(end_i - start_i);
    std::vector<int> labels;
    labels.reserve(end_i - start_i);
    for (int i = start_i; i < end_i; i++)
    {
        int t = data[base_interval]["chart"][i]["time"];
        try
        {
            std::vector<float> param;
            param.reserve(getParamNum());
            for (int i = 0; i < intervals.size(); i++)
            {
                p = getParams();
                param.insert(param.end(), p.begin(), p.end());
            }
            params.push_back(param);
            labels.push_back(getLabel())
        }
        catch (ChartRangeError &e)
        {
            std::cerr << e.what() << '\n';
        }
    }
}

json get_json(string file_path)
{

    std::ifstream ifs(file_path);
    string str;

    if (ifs.fail())
    {
        std::cerr << "Failed to open file." << std::endl;
        std::exit(1);
    }
    json json_obj = json::parse(ifs);
    return json_obj;
}

int main()
{
    std::map<string, json> data;
    for (int i = 0; i < intervals.size(); i++)
    {
        data[intervals[i]] = (get_json("./chart_log/" + symbol + "_" + intervals[i] + ".json"));
    }
    std::cout << data[0]["chart"][0]["time"] << std::endl;
}