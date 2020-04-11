#include "accelerator.h"

#define UNROLLFACTOR 2
#define NUMATT 32562


data_t evalDiscreteAtt_hw(data_t *Stable, data_t *Mtable, data_t *Classes,
                          int length, int width, int offset)
{
    float_t maxGain = -FLT_MAX, MaxGain[] = {-FLT_MAX, -FLT_MAX};
    data_t splittingCol = 0, SplittingCol[] = {0, 0};
    data_t *t[] = {Stable, Mtable};
    int lim[] = {offset, width - 1 - offset};

    data_t classes[NUMATT];
#pragma HLS RESOURCE variable=classes core=RAM_2P_BRAM

    for (int j = 0; j < length; j++)
#pragma HLS LOOP_TRIPCOUNT min=1 max=32562
        classes[j] = Classes[j];

    for (int p = 0; p < UNROLLFACTOR; p++) {
#pragma HLS UNROLL factor=2

        int classFreq[X * Y];
#pragma HLS ARRAY_PARTITION variable=classFreq block factor=16

        int limit = lim[p];
        data_t *table = t[p];
        for (int column = 0; column < limit; column++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=7
            data_t *attribute = table + column * length;
            for (int j = 0; j < X * Y; j++)
#pragma HLS UNROLL
                    classFreq[j] = 0;

            data_t feature = attribute[0];
            for (int i = 1; i < length; i++)
#pragma HLS LOOP_TRIPCOUNT min=1 max=32562
#pragma HLS PIPELINE II=1
                if (UNKNOWN != attribute[i])
                    ++classFreq[attribute[i] + X * classes[i]];

            int items = 0;
            float_t infoGain = 0.0;
            for (data_t i = 0; i < maxAtt; i++) {
#pragma HLS UNROLL factor=4
                int instances = 0;
                for (data_t j = 0; j < maxClass; j++)
#pragma HLS PIPELINE II=1
                    instances += classFreq[i + X * j];

                if (!instances)
                    continue;


                float_t entropy = 0.0;
                for (data_t j = 0; j < maxClass; j++) {
#pragma HLS PIPELINE
                    float_t freq = classFreq[i + X * j];
                    if (freq > 0) {
                        float_t p = freq / instances;
#pragma HLS RESOURCE variable=p core=FDiv
                        float_t logp = log2(p);
                        float_t pEntropy = p * logp;
#pragma HLS RESOURCE variable=pEntropy core=FMul_fulldsp
                        entropy -= pEntropy;
#pragma HLS RESOURCE variable=entropy core=FAddSub_fulldsp
                    }
                }

                items += instances;
                float_t pInfoGain = (float_t) instances * entropy;
#pragma HLS RESOURCE variable=pInfoGain core=FMul_fulldsp
                infoGain -= pInfoGain;
#pragma HLS RESOURCE variable=infoGain core=FAddSub_fulldsp
            }

            infoGain = Div(infoGain, items);
#pragma HLS RESOURCE variable=infoGain core=FDiv


            if (MaxGain[p] <= infoGain) {
                MaxGain[p] = infoGain;
                SplittingCol[p] = feature;
            }
        }

        if (maxGain <= MaxGain[p]) {
            splittingCol = SplittingCol[p];
            maxGain = MaxGain[p];
        }
    }

    return splittingCol;
}
