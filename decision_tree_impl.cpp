#include "accelerator.h"


int attCompare(const void *a, const void *b)
{
    return ((cont_t*) a)->val - ((cont_t*) b)->val;
}

int *evalContinuousAtt(cont_t **contAtt, int length)
{
    data_t numOfClasses = AttCount[Width - 1];
    cont_t *arr = (cont_t*) safe_malloc((length - 1) * sizeof(cont_t));
    int *thresh = (int*) safe_malloc(NumOfContFeatures * sizeof(int));
    int *classFreq = (int*) safe_malloc(PARTS * numOfClasses * sizeof(int));
    int *level[] = {classFreq, classFreq + numOfClasses};

    for (int k = 0, cut; k < NumOfContFeatures; thresh[k++] = cut) {
        int items = ContCount[k];
        memcpy(arr, contAtt[k], items * sizeof(cont_t));
        qsort(arr, items, sizeof(*arr), attCompare);

        memset(classFreq, 0, PARTS * numOfClasses * sizeof(*classFreq));


        for (int i = 0; i < items; i++)
            ++level[1][arr[i].pclass];

        cut = 0;
        float_t maxGain = -FLT_MAX;
        for (int i = 0, j; i < items; i = j) {
            for (j = i; j < items && arr[j].val == arr[i].val; j++) {
                ++level[0][arr[j].pclass];
                --level[1][arr[j].pclass];
            }
            if (j == items)
                break;


            int split = arr[i].val;


            float_t part[PARTS];
            part[0] = (float_t) j;
            part[1] = (float_t) (items - j);

            float_t infoGain = 0.0;
            for (data_t m = 0; m < PARTS; m++) {
                float_t entropy = 0.0;
                for (data_t n = 0; n < numOfClasses; n++) {
                    float_t p = (float_t) level[m][n] / part[m];
                    entropy -= p * Log(p);
                }

                infoGain -= part[m] * entropy;
            }


            if (maxGain <= infoGain) {
                maxGain = infoGain;
                cut = split;
            }
        }
    }

    free(classFreq);
    free(arr);
    return thresh;
}


data_t **mergeData(data_t **trainData, int length,
                   cont_t **contAtt, int *thresh)
{
    for (int j = 0; j < NumOfContFeatures; free(contAtt[j++])) {
        int c = 0, column = ContFeatures[j];
        data_t *feature = trainData[column];
        for (int i = 1; i < length; i++)
            if (feature[i] != UNKNOWN)
                feature[i] = contAtt[j][c++].val > thresh[j];
        Decoding[column].push_back("<=" + to_string(thresh[j]));
        Decoding[column].push_back(">" + to_string(thresh[j]));
    }

    free(contAtt);
    free(ContCount);
    return trainData;
}


void prepareBuild(data_t **trainData, int length, int width)
{
    data_t maxNumOfAtt = 0;
    for (int i = 0; i < width; i++) {
        AttCount[i] = DISCRETE == SpecialStatus[i] ?
                                  sequential(i) : PARTS;
        maxNumOfAtt = AttCount[i] > maxNumOfAtt?
                      AttCount[i] : maxNumOfAtt;
#if DISP_NUM_OF_ATT
        cout << "NumOfAtt[" << i << "]=" << (int)AttCount[i] << endl;
        if (i == width - 1)
            cout << "\tMaxNumOfAtt=" << (int)maxNumOfAtt << endl;
#endif
    }

    AttCount[width] = maxNumOfAtt;

    int lastCol = width - 1;
    int *classFreq = (int*) safe_allocin(AttCount[lastCol],
                                         sizeof(int), 0);

    for (int i = 1; i < length; i++)
        ++classFreq[trainData[lastCol][i]];

    int maxFreq;
    findmax(classFreq, AttCount[lastCol], maxFreq, DefaultClass);

    free(classFreq);
}


node* buildDecisionTree(node* nodePtr,
                        data_t *table, int length, int width)
{
    if (isEmpty(length, width))
        return NULL;

    if (isHomogeneous(table + (width - 1) * length, length)) {
        nodePtr->isLeaf = true;
        nodePtr->label = table[(width - 1) * length + 1];
        nodePtr->bestClass = nodePtr->label;
    } else {
        int offset = width >> 1;
        int _length, _width, maxFreq, numOfClasses = AttCount[Width - 1];
        int *classFreq = (int*) safe_allocin(numOfClasses, sizeof(int), 0);
        data_t *subtable = (data_t*) safe_alloc(sizeof(data_t) *
                                                (width-1) * length);
        data_t splittingCol = evalDiscreteAtt_hw(table, table + offset * length,
                                                 table + length * (width-1),
                                                 length, width, offset);
        nodePtr->splitOn = splittingCol;
        for (data_t i = 0; i < AttCount[splittingCol]; i++) {
            node* newNode = new node();
            newNode->label = i;
            newNode->splitOn = splittingCol;
            pruneRules(subtable, _length = length, _width = width,
                       table, splittingCol, i);
            node* child = buildDecisionTree(newNode,
                                            subtable, _length, _width);
            nodePtr->branch.push_back(child);
            if (child)
                ++classFreq[child->bestClass];
        }
        deallocate(subtable);
        findmax(classFreq, numOfClasses, maxFreq, nodePtr->bestClass);
        free(classFreq);
    }

    return nodePtr;
}


void pruneRules(data_t *subtable, int &length, int &width,
                data_t *table, data_t feature, data_t value)
{
    int column = -1, _length = length, _width = width;


    for (int i = width = 0; i < _width; i++)
        if (table[i * _length] == feature)
            column = i;


    int first = 0 == column;
    for (int i = length = 1; i < _length; i++)
        if (table[column * _length + i] == value)
            subtable[length++] = table[first * _length + i];


    for (int i = width = 0; i < _width; i++)
        if (table[i * _length] != feature)
            subtable[width++ * length] = table[i * _length];


    for (int k = 1, j = 1 + first; j < _width; k += j++ != column)
        if (j != column)
            for (int i = 1, n = 1; i < _length; i++)
                if (table[column * _length + i] == value)
                    subtable[k * length + n++] = table[j * _length + i];
}


data_t evalDiscreteAtt_sw(data_t *table, int length, int width)
{
    float_t maxGain = -FLT_MAX;
    data_t splittingCol = 0, *classes = table + (width - 1) * length;
    data_t numOfClasses = AttCount[classes[0]];

    for (int column = 0; column < width - 1; column++) {
        data_t numOfAtt = AttCount[table[column * length]];
        data_t *attribute = table + column * length;
        int classFreq[numOfAtt][numOfClasses];

        memset(classFreq, 0, sizeof classFreq);

        for (int i = 1; i < length; i++)
            if (UNKNOWN != attribute[i])
                ++classFreq[attribute[i]][classes[i]];

        int items = 0;
        float_t infoGain = 0.0;
        for (data_t i = 0; i < numOfAtt; i++) {
            int instances = 0;
            for (data_t j = 0; j < numOfClasses; j++)
                instances += classFreq[i][j];
            if (!instances)
                continue;


            float_t entropy = 0.0;
            for (data_t j = 0; j < numOfClasses; j++) {
                float_t p = (float_t) classFreq[i][j] / instances;
                entropy -= p * Log(p);
            }

            items += instances;
            infoGain -= (float_t) instances * entropy;
        }

        infoGain = Div(infoGain, items);


        if (maxGain <= infoGain) {
            maxGain = infoGain;
            splittingCol = column;
        }
    }

    return table[splittingCol * length];
}


bool isHomogeneous(data_t *classColumn, int length)
{
    data_t firstValue = classColumn[1];
    for (int i = 1; i < length; i++)
        if (firstValue != classColumn[i])
            return false;
    return true;
}


bool isEmpty(int length, int width)
{


    return length <= 1 || width <= 1;
}


void estimateErrors(node *root, data_t **table,
                    int vstart, int vend, int width)
{
    vdt rule(width);
    for (int i = vstart; i < vend; i++) {
        for (int j = 0; j < width; j++)
            rule[j] = table[j][i];
        testDecisionTree(root, rule, true);
    }
}


void pruneDecisionTree(node *nodePtr)
{
    if (nodePtr == NULL || nodePtr->isLeaf)
        return;
    for (size_t i = 0; i < nodePtr->branch.size(); i++)
        pruneDecisionTree(nodePtr->branch[i]);
    if (nodePtr->error > 0) {
        nodePtr->isLeaf = true;
        nodePtr->label = nodePtr->bestClass;
        destroyBranches(nodePtr);
    }
}


void destroyBranches(node *subtree)
{
    for (size_t i = 0; i < subtree->branch.size(); i++)
        destroyTree(subtree->branch[i]);
    subtree->branch.clear();
}


void destroyTree(node *subtree)
{
    if (subtree) {
        destroyBranches(subtree);
        delete subtree;
    }
}
data_t testDecisionTree(node* root, vdt rule, const bool opt)
{
    node* nodePtr = root;
    node *path[rule.size()];
    data_t prediction = DefaultClass;
    data_t numOfClasses = AttCount[rule.size() - opt];

    int steps = 0;
    while (!nodePtr->branch.empty()) {
        path[steps++] = nodePtr;
        data_t value = rule[nodePtr->splitOn];

        if (UNKNOWN == value) {
#if DEBUG_TEST_MISSING
            cout << "we have missing value @"
            << (int)nodePtr->splitOn << endl;
#endif
            vdt newRule = rule;
            data_t numOfAtt = AttCount[nodePtr->splitOn];
            data_t *classFreq = (data_t*) safe_allocin(numOfClasses,
                                                       sizeof(data_t), 0);
            for (data_t i = 0; i < numOfAtt; i++) {
                newRule[nodePtr->splitOn] = i;
                data_t predClass = testDecisionTree(root, newRule);
                ++classFreq[predClass];
#if DEBUG_TEST_MISSING
                cout << "\tTesting for value of: "
                << decoding(nodePtr->splitOn,
                        newRule[nodePtr->splitOn])
                << " out of " << (int) AttCount[nodePtr->splitOn]
                << " possible ones." << endl
                << "\t->returned: "
                << decoding(rule.size(), predClass) << endl;
#endif
            }
            newRule.clear();

            data_t maxFreq;
            data_t bestClass = DefaultClass;
            findmax(classFreq, numOfClasses, maxFreq, bestClass);
            free(classFreq);

#if DEBUG_TEST_MISSING
            cout << "mostDense: " << decoding(rule.size(), bestClass)
            << " with frequency: " << (int)maxFreq << endl;
#endif

            prediction = bestClass;
            break;
        }

        if (value >= nodePtr->branch.size()) {
            cout << "Branch for value: "
                 << decoding(nodePtr->splitOn, value)
                 << " not found. Is feature "
                 << (int) nodePtr->splitOn
                 << " continuous?" << endl;
            break;
        }

        nodePtr = nodePtr->branch[value];
        if (nodePtr == NULL)
            break;


        if (nodePtr->isLeaf) {
            prediction = nodePtr->label;
            break;
        }
    }

    if (opt) {
        size_t classCol = rule.size() - 1;
        if (prediction != rule[classCol])
            for (int i = 0; i < steps; i++)
                ++path[i]->error;
        for (int i = 0; i < steps; i++) {
            if (path[i]->error == 0)
                path[i]->error += 1e-4;
            if (path[i]->bestClass != rule[classCol])
                --path[i]->error;
        }
    }

    return prediction;
}


float printResults(vdt givenData, vdt predictions, int classCol)
{
    int correct = 0;
#if SAVE_RESULTS
    ofstream outputFile;
    outputFile.open(OUTPUT_FILENAME);

    outputFile << setw(3) << "#" << setw(16) << "Given Class" << setw(31)
               << right << "Predicted Class\n"
               << "--------------------------------------------------\n";
#endif
    for (size_t i = 0; i < givenData.size(); i++) {
#if SAVE_RESULTS
        outputFile << setw(3) << i + 1 << setw(16)
                   << decoding(classCol, givenData[i]);
#endif
        if (givenData[i] == predictions[i]) {
            correct++;
#if SAVE_RESULTS
            outputFile << "  ------------  ";
        } else {
            outputFile << "  xxxxxxxxxxxx  ";
        }
        outputFile << decoding(classCol, predictions[i]) << "\n";
    }

    outputFile << "--------------------------------------------------\n"
               << "Total number of instances in test data = "
               << givenData.size() << "\n"
               << "Number of correctly predicted instances = "
               << correct << "\n";
    outputFile.close();
#else
        }
    }
#endif
    return (float) correct / givenData.size() * 100;
}


void printDecisionTree(node* nodePtr)
{
    if (nodePtr == NULL)
        return;
    if (!nodePtr->branch.empty()) {
        int col = nodePtr->splitOn;
        cout << " Value: "
             << (nodePtr->label >= Decoding[col].size() ?
                              "" : decoding(col, nodePtr->label))
             << "\nSplit @" << (int) nodePtr->splitOn;
        for (size_t i = 0; i < nodePtr->branch.size(); i++) {
            cout << "\t";
            printDecisionTree(nodePtr->branch[i]);
        }
        return;
    } else {
        cout << "Predicted class = "
             << decoding(Width - 1, nodePtr->label) << endl;
        return;
    }
}


void printAttributeTable(data_t **table, int l, int w)
{
    int col, len;
    for (len = 1; len < l; len++)
        for (col = 0; col < w; col++)
            cout << decoding(col, table[col][len])
                 << (col == w - 1 ? "\n" : "\t");
}
