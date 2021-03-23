## I. This repository contains the code for :

### 1. Cross-lingual language model pretraining ([XLM](https://github.com/facebookresearch/XLM)) 

XLM supports multi-GPU and multi-node training, and contains code for:
- **Language model pretraining**:
    - **Causal Language Model** (CLM)
    - **Masked Language Model** (MLM)
    - **Translation Language Model** (TLM)
- **GLUE** fine-tuning
- **XNLI** fine-tuning
- **Supervised / Unsupervised MT** training:
    - Denoising auto-encoder
    - Parallel data training
    - Online back-translation

### 2. Model-Agnostic Meta-Learning ([MAML](https://arxiv.org/abs/1911.02116))  

See [maml](https://github.com/cbfinn/maml), [learn2learn](https://github.com/learnables/learn2learn)...  

See [HowToTrainYourMAMLPytorch](https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch) for a replication of the paper ["How to train your MAML"](https://arxiv.org/abs/1810.09502), along with a replication of the original ["Model Agnostic Meta Learning"](https://arxiv.org/abs/1703.03400) (MAML) paper.

### 3. TRANSFORMERS WITH COMPETITIVE ENSEMBLES OF INDEPENDENT MECHANISMS ([TIM](https://openreview.net/forum?id=1TIrbngpW0x))

### 4. MULTI-TASK LEARNING WITH DEEP NEURAL NETWORKS: A SURVEY ([MTL](https://arxiv.org/pdf/2009.09796.pdf))

## II. Train your own (meta-)model

**Open the illustrative notebook in colab**[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tikquuss/meta_XLM/blob/master/notebooks/demo/tuto.ipynb)

**Note** : Most of the bash scripts used in this repository were written on the windows operating system, and can generate this [error](https://prograide.com/pregunta/5588/configure--bin--sh--m-mauvais-interpreteur) on linux platforms.  
This problem can be corrected with the following command: 
```
filename=my_file.sh 
cat $filename | tr -d '\r' > $filename.new && rm $filename && mv $filename.new $filename 
```
### 1. Preparing the data 

We assume that you have txt files available for preprocessing. Look at the following example for which we have three translation tasks: `English-French, German-English and German-French`.

We have the following files available for preprocessing: 
```
- en-fr.en.txt and en-fr.fr.txt 
- de-en.de.txt and de-en.en.txt 
- de-fr.de.txt and de-fr.fr.txt 
```
All these files must be in the same folder (`PARA_PATH`).  
You can also (only or optionally) have monolingual data available (`en.txt, de.txt and fr.txt`; in `MONO_PATH` folder).  
Parallel and monolingual data can all be in the same folder.

**Note** : Languages must be submitted in alphabetical order (`de-en and not en-de, fr-ru and not ru-fr...`). If you submit them in any order you will have problems loading data during training, because when you run the [train.py](XLM/train.py) script the parameters like the language pair are put back in alphabetical order before being processed. Don't worry about this alphabetical order restriction, XLM for MT is naturally trained to translate sentences in both directions. See [translate.py](scripts/translate.py).

[OPUS collections](http://opus.nlpl.eu/) is a good source of dataset. We illustrate in the [opus.sh](scripts/opus.sh) script how to download the data from opus and convert it to a text file.  
Changing parameters ($PARA_PATH and $SRC) in [opus.sh](scripts/opus.sh).
```
cd meta_XLM
chmod +x ./scripts/opus.sh
./scripts/opus.sh de-fr
```

Another source for `other_languages-english` data is [anki Tab-delimited Bilingual Sentence Pairs](http://www.manythings.org/anki/). Simply download the .zip file, unzip to extract the `other_language.txt` file. This file usually contains data in the form of `sentence_en sentence_other_language other_information` on each line. See [anki.py](scripts/anki.py) and [anky.sh](scripts/anki.sh) in relation to a how to extract data from [anki](http://www.manythings.org/anki/). Example of how to download and extract `de-en` and `en-fr` pair data.
```
cd meta_XLM
output_path=/content/data/para
mkdir $output_path
chmod +x ./scripts/anki.sh
./scripts/anki.sh de,en deu-eng $output_path scripts/anki.py
./scripts/anki.sh en,fr fra-eng $output_path scripts/anki.py
```
After that you will have in `data/para` following files : `de-en.de.txt, de-en.en.txt, deu.txt, deu-eng.zip and _about.txt`  

Move to the `XLM` folder in advance.  
```
cd XLM
```
Install the following dependencies ([fastBPE](https://github.com/facebookresearch/XLM/tree/master/tools#fastbpe) and [Moses](https://github.com/facebookresearch/XLM/tree/master/tools#tokenizers)) if you have not already done so. 
```
git clone https://github.com/moses-smt/mosesdecoder tools/mosesdecoder
git clone https://github.com/glample/fastBPE tools/fastBPE && cd tools/fastBPE && g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
```
  
Changing parameters in [data.sh](data.sh). Between lines 94 and 100 of [data.sh](data.sh), you have two options corresponding to two scripts to execute according to the distribution of the folders containing your data. Option 2 is chosen by default, kindly uncomment the lines corresponding to your option.  
With too many BPE codes (depending on the size of the dataset) you may get this [error](https://github.com/glample/fastBPE/issues/7). Decrease the number of codes (e.g. you can dichotomously search for the appropriate/maximum number of codes that make the error disappear)

```
languages=de,en,fr
chmod +x ../data.sh 
../data.sh $languages
```

If you stop the execution when processing is being done on a file please delete this erroneous file before continuing or restarting the processing, otherwise the processing will continue with this erroneous file and potential errors will certainly occur.  

After this you will have the following (necessary) files in `$OUTPATH` (and `$OUTPATH/fine_tune` depending on the parameter `$sub_task`):  

```
- monolingual data :
    - training data   : train.fr.pth, train.en.pth and train.de.pth
    - test data       : test.fr.pth, test.en.pth and test.de.pth
    - validation data : valid.fr.pth, valid.en.pth and valid.de.pth
- parallel data :
    - training data : 
        - train.en-fr.en.pth and train.en-fr.fr.pth 
        - train.de-en.en.pth and train.de-en.de.pth
        - train.de-fr.de.pth and train.de-fr.fr.pth 
    - test data :
        - test.en-fr.en.pth and test.en-fr.fr.pth 
        - test.de-en.en.pth and test.de-en.de.pth
        - test.de-fr.de.pth and test.de-fr.fr.pth 
    - validation data
        - valid.en-fr.en.pth and valid.en-fr.fr.pth 
        - valid.de-en.en.pth and valid.de-en.de.pth
        - valid.de-fr.de.pth and valid.de-fr.fr.pth 
 - code and vocab
```

### 2. Pretrain a language (meta-)model 

Install the following dependencie ([Apex](https://github.com/nvidia/apex#quick-start)) if you have not already done so.
```
git clone https://github.com/NVIDIA/apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
```

Instead of passing all the parameters of train.py, put them in a json file and specify the path to this file in parameter (See [lm_template.json](configs/lm_template.json) file for more details).
```
config_file=../configs/lm_template.json
python train.py --config_file $config_file
```
If you pass a parameter by calling the script [train.py](XLM/train.py) (example: `python train.py --config_file $config_file --data_path my/data_path`), it will overwrite the one passed in `$config_file`.  
Once the training is finished you will see a file named `train.log` in the `$dump_path/$exp_name/$exp_id` folder information about the training. You will find in this same folder your checkpoints and best model.  
When `"mlm_steps":"..."`, train.py automatically uses the languages to have `"mlm_steps":"de,en,fr,de-en,de-fe,en-fr"` (give a precise value to mlm_steps if you don't want to do all MLM and TLM, example : `"mlm_steps":"en,fr,en-fr"`). This also applies to `"clm_steps":"..."` which deviates `"clm_steps":"de,en,fr"` in this case.    

Note :  
-`en` means MLM on `en`, and requires the following three files in `data_path`: `a.en.pth, a ∈ {train, test, valid} (monolingual data)`  
-`en-fr` means TLM on `en and fr`, and requires the following six files in `data_path`: `a.en-fr.b.pth, a ∈ {train, test, valid} and b ∈ {en, fr} (parallel data)`  
-`en,fr,en-fr` means MLM+TLM on `en, fr, en and fr`, and requires the following twelve files in `data_path`: `a.b.pth and a.en-fr.b.pth, a ∈ {train, test, valid} and b ∈ {en, fr}`  

To [train with multiple GPUs](https://github.com/facebookresearch/XLM#how-can-i-run-experiments-on-multiple-gpus) use:
```
export NGPU=8; python -m torch.distributed.launch --nproc_per_node=$NGPU train.py --config_file $config_file
```

**Tips**: Even when the validation perplexity plateaus, keep training your model. The larger the batch size the better (so using multiple GPUs will improve performance). Tuning the learning rate (e.g. [0.0001, 0.0002]) should help.

In the case of <b>metalearning</b>, you just have to specify your meta-task separated by `|` in `lgs` and `objectives (clm_steps, mlm_steps, ae_steps, mt_steps, bt_steps and pc_steps)`.  
For example, if you only want to do metalearning (without doing XLM) in our case, you have to specify these parameters: `"lgs":"de-en|de-fr|en-fr"`, `"clm_steps":"...|...|..."` and/or `"mlm_steps":"...|...|..."`. These last two parameters, if specified as such, will become respectively `"clm_steps":"de,en|de,fr|en,fr"` and/or `"mlm_steps":"de,en,de-en|de,fr,de-fr|en,fr,en-fr"`.  
The passage of the three points follows the same logic as above. That is to say that if at the level of the meta-task `de-en`:  
	- we only want to do MLM (without TLM): `mlm_steps` becomes `"mlm_steps": "de,en|...|..."`  
	- we don't want to do anything: `mlm_steps` becomes `"mlm_steps":"|...|..."`.

It is also not allowed to specify a meta-task that has no objective. In our case, `"clm_steps":"...||..."` and/or `"mlm_steps":"...||..."` will generate an exception, in which case the meta-task `de-fr` (second task) has no objective.

If you want to do metalearning and XLM simultaneously : 
- `"lgs":"de-en-fr|de-en-fr|de-en-fr"` 
- Follow the same logic as described above for the other parameters.

###### Description of some essential parameters

```
## main parameters
exp_name                     # experiment name
exp_id                       # Experiment ID
dump_path                    # where to store the experiment (the model will be stored in $dump_path/$exp_name/$exp_id)

## data location / training objective
data_path                    # data location 
lgs                          # considered languages/meta-tasks
clm_steps                    # CLM objective
mlm_steps                    # MLM objective

## transformer parameters
emb_dim                      # embeddings / model dimension
n_layers                     # number of layers
n_heads                      # number of heads
dropout                      # dropout
attention_dropout            # attention dropout
gelu_activation              # GELU instead of ReLU

## optimization
batch_size                   # sequences per batch
bptt                         # sequences length
optimizer                    # optimizer
epoch_size                   # number of sentences per epoch
max_epoch                    # Maximum epoch size
validation_metrics           # validation metric (when to save the best model)
stopping_criterion           # end experiment if stopping criterion does not improve

## dataset
#### These three parameters will always be rounded to an integer number of batches, so don't be surprised if you see different values than the ones provided.
train_n_samples              # Just consider train_n_sample train data
valid_n_samples              # Just consider valid_n_sample validation data 
test_n_samples               # Just consider test_n_sample test data for
#### If you don't have enough RAM/GPU or swap memory, leave these three parameters to True, otherwise you may get an error like this when evaluating :
###### RuntimeError: copy_if failed to synchronize: cudaErrorAssert: device-side assert triggered
remove_long_sentences_train # remove long sentences in train dataset      
remove_long_sentences_valid # remove long sentences in valid dataset  
remove_long_sentences_test  # remove long sentences in test dataset  

tim_layers_pos # position of TIM layers in encoder layer (example : 0,1)
n_s # number of mechanisms
dim_feedforward # hidden dimension of feed forward neural net
log_interval # Interval (number of steps) between two displays : batch_size by default
device  # device name (cuda / cpu)
random_seed # random seed for reproductibility
```

###### There are other parameters that are not specified here (see [params.py](XLM/params.py))

### 3. Train a (unsupervised/supervised) MT from a pretrained meta-model 

See [mt_template.json](configs/mt_template.json) file for more details.
```
config_file=../configs/mt_template.json
python train.py --config_file $config_file
```

When the `ae_steps` and `bt_steps` objects alone are specified, this is unsupervised machine translation, and only requires monolingual data. If the parallel data is available, give `mt_step` a value based on the language pairs for which the data is available.  
All comments made above about parameter passing and <b>metalearning</b> remain valid here : if you want to exclude a meta-task in an objective, put a blank in its place. Suppose, in the case of <b>metalearning</b>, we want to exclude from `"ae_steps":"en,fr|en,de|de,fr"` the meta-task:
- `de-en` : `ae_steps`  becomes `"ae_steps":"en,fr||de,fr"` 
- `de-fr` : `ae_steps`  becomes `"ae_steps":"en,fr|de,en|"`  

###### Description of some essential parameters  
The description made above remains valid here
```
## main parameters
reload_model     # model to reload for encoder,decoder
## data location / training objective
ae_steps          # denoising auto-encoder training steps
bt_steps          # back-translation steps
mt_steps          # parallel training steps
word_shuffle      # noise for auto-encoding loss
word_dropout      # noise for auto-encoding loss
word_blank        # noise for auto-encoding loss
lambda_ae         # scheduling on the auto-encoding coefficient

## transformer parameters
encoder_only      # use a decoder for MT

## optimization
tokens_per_batch  # use batches with a fixed number of words
eval_bleu         # also evaluate the BLEU score
```
###### There are other parameters that are not specified here (see [params.py](XLM/params.py))

### 4. Fine-tune a pretrained meta-model on classification task

Download the bias_data.csv file [here](https://drive.google.com/file/d/1S6R6ckjNe7TDTwela2o-y3Fabl8lcq0H/view?usp=sharing)

```
DATA_PATH=/content/classification_data
python ../split_data.py -d /content/bias_data.csv -o $DATA_PATH  -v 0.2 -r 0 -t classification
```

See [classif_template.json](configs/classif_template.json) file for more details.
```
config_file=../configs/classif_template.json
python classify.py --config_file $config_file --train_data_file $DATA_PATH/bias_data_train.csv  --val_data_file $DATA_PATH/bias_data_val.csv
```

### 5. case of metalearning : optionally fine-tune the meta-model on a specific (sub) nmt (meta) task 

At this point, if your fine-tuning data did not come from the previous pre-processing, you can just prepare your txt data and call the script build_meta_data.sh with the (sub) task in question. Since the codes and vocabulary must be preserved, we have prepared another script ([build_fine_tune_data.sh](scripts/build_fine_tune_data.sh)) in which we directly apply BPE tokenization on dataset and binarize everything using preprocess.py based on the codes and vocabulary of the meta-model. So we have to call this script for each subtask like this :

```
languages = 
chmod +x ../ft_data.sh
../ft_data.sh $languages
```

At this stage, restart the training as in the previous section with :
- lgs="en-fr"
- reload_model = path to the folder where you stored the meta-model
- `bt_steps'':"..."`, `ae_steps'':"..."` and/or `mt_steps'':"..."` (replace the three bullet points with your specific objectives if any)  
You can use one of the two previously trained meta-models: pre-formed meta-model (MLM, TLM) or meta-MT formed from the pre-formed meta-model. 
