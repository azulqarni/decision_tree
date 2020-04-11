#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <climits>
#include <cfloat>
#include <cmath>
#include <ctime>

using namespace std;

typedef float float_t;
typedef unsigned char data_t;
typedef vector<data_t> vdt;
typedef vector<vdt> vvdt;
typedef vector<string> vs;
typedef vector<vs> vvs;
typedef map<string, data_t> msdt;
typedef vector<msdt> vmsdt;
typedef int (*integer)(string);

#define allocate sds_alloc
#ifndef allocate
#define allocate malloc
#define deallocate free
#elif allocate == sds_alloc
#define deallocate sds_free
#include <sds_lib.h>
#endif

#ifndef DEBUG
#define DEBUG 0
#endif
#ifndef DEBUG_TEST_MISSING
#define DEBUG_TEST_MISSING 0
#endif
#ifndef DISP_NUM_OF_ATT
#define DISP_NUM_OF_ATT 0
#endif
#ifndef IMPURE_INPUT
#define IMPURE_INPUT 0
#endif
#ifndef SAVE_RESULTS
#define SAVE_RESULTS 1
#endif

#define UNKNOWN (data_t) -1
#define DISCRETE 0xFF
#define PARTS 2
#define BASE_WIDTH 8
#define BASE_LENGTH 512
#define OUTPUT_FILENAME "predictions.txt"
#define Log(x) ((x) <= 0 ? 0.0 : log2(x))
#define Div(x,y) (((y) == 0) ? -66.6 : (x) / (y))

#define skipLine(f) (f).ignore(numeric_limits<streamsize>::max(), '\n');

#define purify(s) \
        (s).erase( \
            remove_if((s).begin(), (s).end(), \
                [](char c) {return isblank(c) || c == '.';}), \
            (s).end()); \
        if (0 == (s).length()) \
            return;
			
#define lengthof(p) sizeof(p) / sizeof(*p)

#define free_2d(p,len) \
    for(int __the_var = 0; \
            __the_var < len; \
            __the_var ++) \
        free(p[__the_var]);
		
#define findmax(vector,len,max,idx) \
    max = 0; \
    for (data_t __the_var = 0; \
                __the_var < len; \
                __the_var ++) \
        if (vector[__the_var] > max) \
            max = vector[__the_var], \
            idx = __the_var;
			
struct node {
    bool isLeaf;
    data_t splitOn;
    data_t label;
    data_t bestClass;
    float error;
    vector<node*> branch;
};

struct table_t {
    int length;
    int width;
    data_t **att;
};

struct cont_t {
    int val;
    data_t pclass;
};

extern vvs Decoding;
extern vmsdt Encoding;
extern data_t *AttCount;
extern unsigned char *SpecialStatus;
extern float XvalRatio;
extern int *ContCount;
extern int *ContFeatures;
extern int NumOfContFeatures;
extern int DefaultClass;
extern int Width, Length;
void *safe_malloc(size_t);
void *safe_alloc(size_t);
void *reshape(void*, size_t);
data_t **safe_malloc_2d_dt(int, int);
cont_t **safe_malloc_2d_ct(int, int);
void *safe_allocin(int, size_t, const unsigned char);
cont_t **initParameters(string);
data_t **initStructures(int);
int splitValidTrain(data_t* &, data_t**, int, int);
void skipHeader(ifstream&, string);
data_t sequential(int);
data_t encoding(int, string);
string decoding(int, data_t);
void parseTrain(string, int&, data_t**, cont_t**);
void parseTest(string, vvdt&, vdt&, int*);
int attCompare(const void *, const void *);
int *evalContinuousAtt(cont_t**, int);
data_t **mergeData(data_t**, int, cont_t**, int*);
void prepareBuild(data_t**, int, int);
node* buildDecisionTree(node*, data_t*, int, int);
void pruneRules(data_t*, int&, int&, data_t*, data_t, data_t);
data_t evalDiscreteAtt_sw(data_t*, int, int);
bool isHomogeneous(data_t*, int);
bool isEmpty(int, int);
void estimateErrors(node*, data_t**, int, int, int);
void destroyBranches(node*);
void destroyTree(node*);
void pruneDecisionTree(node*);
data_t testDecisionTree(node*, vdt, const bool = false);
float printResults(vdt, vdt, int);
void printDecisionTree(node*);
void printAttributeTable(data_t**, int, int);
