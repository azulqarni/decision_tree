#include "decision_tree_def.h"

int main(int argc, const char *argv[]) {

    ifstream inputFile;

    string singleLine, file[2];

    for (int i = 0; i < 2; i++)
        file[i] = argc <= 1 + i ? "" : argv[1 + i];

    cont_t **contAtt = initParameters(string(file[0]));

    inputFile.open(file[0]);
    if (!inputFile) {

        cerr << "Error: Training data file not found!\n";
        exit(-1);
    }
    skipHeader(inputFile, string(argv[1]));
    getline(inputFile, singleLine);


    int length = 1;
    int width = count(singleLine.begin(), singleLine.end(), ',');
    data_t **trainData = initStructures(++width);

    cout << "Loading data..." << endl;
    clock_t t = clock();


    do {
        parseTrain(singleLine, length, trainData, contAtt);
    } while (getline(inputFile, singleLine));

    t = clock() - t;
    double time_taken = ((double) t) / CLOCKS_PER_SEC;
    cout << "Data loaded in " << time_taken << "s" << endl;


    inputFile.close();
    prepareBuild(trainData, length, width);

    cout << "Evaluating continuous data (" << NumOfContFeatures
         << " continuous features)..." << endl;
    t = clock();

    int *thresh = evalContinuousAtt(contAtt, length);


    trainData = mergeData(trainData, length, contAtt, thresh);

    t = clock() - t;
    time_taken = ((double) t) / CLOCKS_PER_SEC;
    cout << "Data evaluated in " << time_taken << "s" << endl;


    node* root = new node();

    data_t *newTrainData;
    int valStart = splitValidTrain(newTrainData, trainData, length, width);

    cout << "Building Tree..." << endl;
    t = clock();


    root = buildDecisionTree(root, newTrainData, valStart, width);

    t = clock() - t;
    time_taken = ((double) t) / CLOCKS_PER_SEC;
    cout << "Tree built in " << time_taken << "s" << endl;


    deallocate(newTrainData);

#if DEBUG
    cout << "\n########### Tree ###########\n";
    printDecisionTree(root);
    cout << "\n######### Tree End #########\n";
#endif

    cout << "Pruning Tree..." << endl;
    t = clock();

    estimateErrors(root, trainData, valStart, length, width);


    pruneDecisionTree(root);

    t = clock() - t;
    time_taken = ((double) t) / CLOCKS_PER_SEC;
    cout << "Tree pruned in " << time_taken << "s" << endl;


    free_2d(trainData, width);
    inputFile.clear();


    vvdt testData;
    vdt classLabels;


    inputFile.open(file[1]);
    if (!inputFile) {

        cerr << "Error: Testing data file not found!\n";
        exit(-1);
    }

    skipHeader(inputFile, string(argv[2]));
    while (getline(inputFile, singleLine))
        parseTest(singleLine, testData, classLabels, thresh);

    inputFile.close();

    free(thresh);
    free(SpecialStatus);
    Encoding.clear();


    vdt predClassLabels;

    cout << "Testing data..." << endl;
    t = clock();

    for (size_t i = 0; i < testData.size(); i++) {

        data_t att = testDecisionTree(root, testData[i]);
        predClassLabels.push_back(att);
    }

    t = clock() - t;
    time_taken = ((double) t) / CLOCKS_PER_SEC;
    cout << "Data tested in " << time_taken << "s" << endl;

    free(AttCount);
    testData.clear();

#if SAVE_RESULTS
    ofstream outputFile;
    outputFile.open(OUTPUT_FILENAME, ios::app);
    outputFile << "\n--------------------------------------------------\n";
#endif

    float accuracy = printResults(classLabels, predClassLabels, width - 1);
    Decoding.clear();
    classLabels.clear();
    predClassLabels.clear();

#if SAVE_RESULTS
    outputFile << "Accuracy of decision tree classifier = " << accuracy
               << "%\n";
#endif

    cout << "Done. ";
    cout << "Accuracy of decision tree classifier = " << accuracy << "%\n";
    return 0;
}
