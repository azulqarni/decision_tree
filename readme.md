# Decision Tree Classifier
Decision Tree Classification variants are among the most popular machine learning algorithms and have been applied in various fields  with success. Their versatility and popularity along with the ease to use make it imperative that solutions be found regarding its performance optimization problem, hence in this paper we tackle this issue by applying methods to optimize a Decision Tree Learning implementation (version C4.5), that will be executed in a heterogeneous computing system involving FPGA along with CPU, making use of the tools
offered by the Software-Defined System On Chip (SDSoC v.2018.3) development platform.


# Publication
If you use any part of this work, please cite the following paper:

- Hardware Acceleration of Decision Tree Learning Algorithm, Asim Zoulkarni, Christoforos Kachris, and Dimitrios Soudris. International Conference on Modern Circuits and Systems Technologies (MOCAST), 2020.


### Running a DTC example in a heterogeneous system involving FPGA
The project folder includes the source code of the Decision Tree Classifier in C/C++ including Xilinx SDSoC directives for hardware implementation and an example data set (Adult).

`!The code of the hardware function is not fully annotated and contains only interface directives.!`

If you want to create a SDSoC project using these sources you may find the following instructions helpful:

1.  Launch SDSoC and create a new empty project. Choose `zed` as target platform.
1.  Add the C/C++ sources in `src/` and set `evalDiscreteAtt_hw()` as hardware function. Set clock frequency at `142.86 MHz`.
1.  All design parameters are set in the file `src/accelerator.h`.
1.  Select `Generate Bitstream` and `Generate SD Card Image`.
1.  Run `SDRelease`.


#### Performance
speed-up (vs ARM Cortex-A9)                         |   2.48
:---------------------------------------------------|----------:
SW-only `ARM Cortex-A9 @ 666.67MHz` (Measured time) |   2.33s
HW accelerated (Measured time)                      |   .94s


#### Resource utilization estimates for hardware accelerator
Resource    |   Used    |   Total   |   % Utilization
:----------:|----------:|----------:|:----------
DSP         |   48      |   220     |   21.82
BRAM        |   75      |   140     |   53.57
LUT         |   35807   |   53200   |   67.31
FF          |   47608   |   106400  |   44.74


### Contacts
For any question, please contact the authors:

* Christoforos Kachris: kachris@microlab.ntua.gr
* Asim Zoulkarni: el13068@mail.ntua.gr


