

# Install packages

    $ conda create --name cscores --file requirements.txt  
    $ source activate cscores
    $ pip install editdistance 

# Configure 

Edit the `config` file, you should set the kaldi alignement phone file for the
language you are evaluating

# exaple of use 

In the test directory I include three files:

- mandarin.phn: the gold phoneme alignments
- mandarin.txt: the baseline of ZRTools used in the Challenge 2017 in the 
                tde evaluation format.
- mandarin.out: the raw output of ZRTools discorery for the Challenge 2017, in the
                ZRTools format.

the script use to compute the ned for a file with the class tde format (mandarin.txt) is 
`compute_ned.py`, for example for the test mandarin.txt file you can do:

    $ . config
    $ ./compute_ned.py test/mandarin.txt | tee mandarin.log

to decode the log file I use:

    $ sed -rn '/^.*within: NED=(.*) std=.*pairs=(.*)$/\1 \2/p' mandarin.log

to compute ned from the raw output of ZRTools I do:

    $ ./zrtools2eval.py test/mandarin.out > mandarin.class 
    $ ./compute_ned.py mandarin.class | tee mandarin.log


