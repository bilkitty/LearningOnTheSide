$PROJDIR=/home/bilkit/Workspace/SideProjs
$TESTDIR=$PROJDIR/test

#
# Env Setup
#


Append to ~/.bashrc
  export PYTHONPATH="$PYTHONPATH:$PROJDIR/common:$PROJDIR/algo:$PROJDIR/gym"

$ source ~/.bashrc

#
# Required Python packages
#


python version=Python 3.7.3
  gym
  numpy
  matplotlib
  decorator
  tkinter

# Commandline  
$ python3 -m pip install gym, numpy, decorator, matplotlib
$ sudo apt-get install python3.6-tk 


#
# Running Tests 
#


All unit tests are organized by proj under $TESTDIR. Test data are contained in $TESTDIR/data.


# Commandline
$ cd $TESTDIR/ 
$ python3 -m unittest discover -p *unittests.py


