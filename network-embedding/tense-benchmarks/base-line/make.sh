g++ init.cpp -o init.so -fPIC -shared -pthread -O3 -march=native
rm -r bin 2>/dev/null
python -m compileall -l . 
mkdir bin
mv *.pyc  bin

