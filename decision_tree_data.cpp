#include "decision_tree_def.h"

vvs Decoding;
vmsdt Encoding;
data_t *AttCount;
unsigned char *SpecialStatus;
float XvalRatio;
int *ContCount;
int *ContFeatures;
int NumOfContFeatures;
int DefaultClass;
int Width;
integer *toint;

int toint_default(string str)
{
    return stoi(str);
}

int toint_alt(string str)
{

    return (int) 100 * stof(str);
}


cont_t** initParameters(string filename)
{
    const size_t last_slash_idx = filename.find_last_of("\\/");
    if (std::string::npos != last_slash_idx)
        filename.erase(0, last_slash_idx + 1);


    if (filename == "adult.all" || filename == "adult.data") {
        static int contFeatures[] = { 1, 3, 5, 11, 12, 13 };
        NumOfContFeatures = lengthof(contFeatures);
        ContFeatures = contFeatures;
        XvalRatio = .73;
    } else if (filename == "census-income.data") {
        static int contFeatures[] = { 1, 6, 17, 18, 19, -25, 40 };
        NumOfContFeatures = lengthof(contFeatures);
        ContFeatures = contFeatures;
        XvalRatio = .97;
    } else if (filename == "train.dat") {
        static int contFeatures[] = { };
        NumOfContFeatures = lengthof(contFeatures);
        ContFeatures = contFeatures;
        XvalRatio = .75;
    } else if (filename == "covtype.train") {
        static int contFeatures[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        NumOfContFeatures = lengthof(contFeatures);
        ContFeatures = contFeatures;
        XvalRatio = .5;
    } else if (filename == "some_future_filename") {
        static int contFeatures[] = { 1, 3, 5 };
        NumOfContFeatures = lengthof(contFeatures);
        ContFeatures = contFeatures;
        XvalRatio = .9;
    } else {
        static int contFeatures[] = { };
        NumOfContFeatures = lengthof(contFeatures);
        ContFeatures = contFeatures;
        XvalRatio = 1;
    }

    int size = BASE_LENGTH;
    toint = (integer*) safe_malloc(NumOfContFeatures * sizeof(integer));
    ContCount = (int*) safe_allocin(NumOfContFeatures, sizeof(int), 0);
    cont_t **contAtt = safe_malloc_2d_ct(NumOfContFeatures, size);

    return contAtt;
}


data_t **initStructures(int width)
{
    int size = BASE_LENGTH;
    data_t **trainData = safe_malloc_2d_dt(width, size);
    for (int i = 0; i < width; i++)
        trainData[i][0] = i;

    Width = width;
    Decoding.resize(width);
    Encoding.resize(width);

    AttCount = (data_t*) safe_malloc((1 + width) * sizeof(data_t));
    SpecialStatus = (unsigned char*) safe_allocin(width,
                                                  sizeof(unsigned char),
                                                  DISCRETE);

    for (unsigned char i = 0; i < NumOfContFeatures; i++) {
        if (ContFeatures[i] > 0) {
            SpecialStatus[--ContFeatures[i]] = i;
            toint[i] = toint_default;
        } else {
            SpecialStatus[ContFeatures[i] ^= -1] = i;
            toint[i] = toint_alt;
        }
    }

    return trainData;
}


int splitValidTrain(data_t* &newTrain, data_t **trainData,
                    int length, int width)
{
    int valStart = (int) length * XvalRatio;

    newTrain = (data_t*) safe_alloc(sizeof(data_t) * valStart * width);

    for (int j = 0; j < width; j++)
        memcpy(newTrain + j * valStart, trainData[j],
               valStart * sizeof(data_t));

    return valStart;
}


void skipHeader(ifstream &infile, string filename)
{


    vs filesList = { "train.dat", "test.dat", "adult.test" };

    const size_t last_slash_idx = filename.find_last_of("\\/");
    if (string::npos != last_slash_idx)
        filename.erase(0, last_slash_idx + 1);

    for (size_t i = 0; i < filesList.size(); i++)
        if (filename == filesList[i]) {
            skipLine(infile);
            break;
        }

    filesList.clear();
}


data_t sequential(int column)
{
    static bool *warning = (bool*) safe_allocin(Width, sizeof(bool), 0);
    static data_t *seq = (data_t*) safe_allocin(Width, sizeof(data_t), -1);
    if (0 == ++seq[column]) {


        if (warning[column]) {
            cerr << "Error: Overflow occurred - "
                    "Make sure you have used sufficient number of "
                    "bits for attribute encoding." << endl;
            exit(1);
        } else
            warning[column] = true;
    }
    return seq[column];
}


data_t encoding(int column, string att)
{
    msdt::iterator it;
    it = Encoding[column].find(att);
    if (it == Encoding[column].end()) {
        data_t seq;
        if ("?" == att)
            seq = UNKNOWN;
        else {
            seq = sequential(column);
            Decoding[column].push_back(att);
        }
        Encoding[column][att] = seq;
        return seq;
    } else
        return it->second;
    return -1;
}


string decoding(int column, data_t value)
{
    return UNKNOWN == value ?
           "?" : Decoding[column][value];
}


void parseTrain(string rule, int &line,
                data_t **table, cont_t **contAtt)
{
#if IMPURE_INPUT
    purify(rule);
#endif

    static int size = BASE_LENGTH;

    if (line == size) {
        size <<= 1;
        for (int i = 0; i < Width; i++)
            table[i] = (data_t*) reshape(table[i],
                                         size * sizeof(data_t));
        for (int i = 0; i < NumOfContFeatures; i++)
            contAtt[i] = (cont_t*) reshape(contAtt[i],
                                           size * sizeof(cont_t));
    }

    int column;
    for (column = 0; rule.find(',') != string::npos; column++) {
        size_t pos = rule.find_first_of(',');
        string att = rule.substr(0, pos);
        unsigned char i = SpecialStatus[column];
        if (DISCRETE == i)
            table[column][line] = encoding(column, att);
        else {
            if ("?" == att)
                table[column][line] = UNKNOWN;
            else {
                contAtt[i][ContCount[i]].val = toint[i](att);

                table[column][line] = 0;
            }
        }
        rule.erase(0, pos + 1);
    }


    data_t classEnc = encoding(column, rule);
    table[column][line] = classEnc;
    for (int i = 0; i < NumOfContFeatures; i++)
        contAtt[i][ContCount[i]++].pclass = classEnc;
    ++line;
}


void parseTest(string rule, vvdt &table,
               vdt &classLabels, int *thresh)
{
#if IMPURE_INPUT
    purify(rule);
#endif

    int column;
    vdt vectorOfAtt(Width - 1);
    for (column = 0; rule.find(',') != string::npos; column++) {
        size_t pos = rule.find_first_of(',');
        string att = rule.substr(0, pos);
        unsigned char i = SpecialStatus[column];
        data_t val = DISCRETE == i ? encoding(column, att) :
                        "?" == att ? UNKNOWN : toint[i](att) > thresh[i];
        vectorOfAtt[column] = val;
        rule.erase(0, pos + 1);
    }


    classLabels.push_back(encoding(column, rule));
    table.push_back(vectorOfAtt);
    vectorOfAtt.clear();
}
