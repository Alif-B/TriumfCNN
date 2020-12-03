# TriumfCNN
Documentation

cProfile

ncalls specifies the number of times the code was called

tottime is the total time spent on executing this function and it's sub functions

percall is tottime divided by ncalls to calculate time spent on each calls average

cumtime is cumulative time spent on the function including sub fuctions AND recursive functions

percall is cumtime divided by primitive calls

Profile funciton is tied to pr variable, which is used to start and stop profiler capture to specify which function to capture, in this case, is GenerateMultiMuonSample_h5. After the capture, the output is saved as test.txt, which can be used to see which code is causing the delay.

In the GenerateMultiMuonSample.py. the output of the profiler is saved as test.txt, which is easy to read and analyze.

To run profiler outside of the python file, simply run the profile.bat with the file you wish to run the profiler with, like so:
profile.bat GenerateMultiMuonSample.py
