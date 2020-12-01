# TriumfCNN
Documentation
- cProfile
ncalls
    Specifies the number of times the code was called

tottime
    Total time spent on executing this function and it's sub functions

percall
    tottime divided by ncalls to calculate time spent on each calls average

cumtime
    Cumulative Time spent on the function including sub fuctions AND recursive functions

percall
    cumtime divided by primitive calls

Profile funciton is tied to pr variable, which is used to start and stop profiler capture to specify which function to capture, in this case, is GenerateMultiMuonSample_h5. After the capture, the output is saved as test.txt, which can be used to see which code is causing the delay
