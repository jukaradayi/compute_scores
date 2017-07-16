

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

the use is like, for mandarin.txt:

    $ . config
    $ python test/mandarin.txt



it will output a log that you will need to decode  

