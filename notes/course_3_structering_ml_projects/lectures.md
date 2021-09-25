# Course 3: Structering Machine Learning Projects

## Week 1

## Intrucduction to ML Strategy

### Why ML Startegy

![motivation](2021-02-07-14-25-20.png)

There are many ways to improve DL system. needes quick and effective ways to understand best strategy to improve.ways of analyzing problems that will point us in the direction of most promising things to try and which ideas we can discard.

### Orthogonalization

![tv_example](2021-02-07-14-33-15.png)

What is orthogonalization process? Understand what one knob to tune in order to achieve one effect. (old TV knobs example)

![chain](2021-02-07-15-23-04.png)

- You can thinkg of the process as a chain, you should not try to fix a link before all previous links works fine.
- links:
  - training: bigger network, Adam
  - dev: regularization, bigger train set
  - test: bigger dev set
  - change dev set or cost function
- why not to use early stopping? one knob that affect both train & dev

## Setting up your goal

### Single number evaluation metric

![single](2021-02-07-16-28-06.png)

- progress is much faster if you have a single real number evaluation metric
- Example #1: in many cases precision/recall are a tradeoff. rather than 2 evaluation metric you better choose one metric, in that case you can choose F1 (or precision or recall).
- Example #2: Instead tracking metric performance over several countries, it better to average it and use it as a single number metric

![example_2](2021-02-07-16-50-34.png)

### Satisficing and Optimizing metrics

![examples](2021-02-08-17-48-37.png)

- if you have N mertics In many cases it reasonable to choose one of them to optimze for and use the other as satisfactory criterias
- Example 1: maximize cat classification accuracy subjet to model r
unning time is under 100ms
- Example 2: trigger word detection (siri, alexa "wake up") maximum accuracy subject to one false positive trigger at most.

### Train/dev/test distribution

![dev_test_set](2021-02-08-17-54-05.png)

- Why spliting dev/test sey by countries is a bad idea? coming from different distribution
- as an analog to dev set + setting a metric we can imagine target where where to shoot arrows. but if test set coming from different distribution you should target your arrows to different place.
- a better idea is to shuffle all your data and split it randonly

![loan_example](2021-02-08-17-55-59.png)

- Spent three month to discover that their model is worth nothing on their target product data (which come from low income zip)

#### Guideline

Choose a devb set and test set to reflect data you expect to get in the future and consider important to do well on.
and again, dev/test should come from the same distribution.

### Size of the dev and test sets

- In the past the rule of thumb was 70/30 train/test or 60/20/20 train/dev/test
- This rule is suite to the past era where we had 100-10000 samples
- In the big data era, we have millions of samples.
- In that case 98/1/1 is more reasonable split.
- Size of test set should be big enouhj to give high confidence in the overall performance of your system. 10k-100k
- In case confidence is not necessary you can choose no test at all just train/dev (Andrew doesn't recommend that).

### When to change dev/test sets and metrics

![metric_change](2021-02-08-18-07-51.png)

- algorithm A have better metric results, but serve pornographic.
- mertic + dev prefer A, you/user prefer B.
- The problem is porn/not porn doesn't calculated in the metric.
- One way to change the metric is add bigger penelty weight (x10) for porn images (needs to label porn/not porn)

![orthogonalization](2021-02-08-18-09-51.png)

![example_2](2021-02-08-18-14-46.png)

- Another example: after deploying model you find that the better model on dev/test get lower metrics on the production images. In that case the reason is high quality images on the train/dev/test data sets and user images on the production (different distribution)
- Needs to change your metric and/or dev/test set.

## Comparing to human-level performance

### Why human-level performance?

1. It become much more feasible in a lot of application areas for machine learning algorithms to become comperatitve with human-level performance.
2. The workflow is much more efficient when you're trying to do something that humans can also do.

What is the bayesoptimal error? best optimal error, there is no way for any function from x to y to supreass a certian level of accuracy.
for example, some image are so blurry that it is not possible to decide if it is cat or not. another example is audio recovery.

![hm](2021-02-08-22-14-08.png)

why it is not so far from human-level performance?

1. In many applications human-level performance is not so far from the bayes optimal error.
2. When error is worse than human-level you have more tools to improve performance that are harder to use once you've suprassed human-level performance.

What tools can be used when ML is worse than humans?

- get labeled data from humans
- gain insight from manual error analysis: why did a person get this right?
- better analysis of bias/variance

### Aviodable bias

![aviodable_bias](2021-02-08-22-32-42.png)

- what should you do if the gap of training error to human error is hugh? (train worse)
- focus on bias, reduce the train error

- what should you do if the error of train is slightly worse than human-level?
- focus on variance, reduce the gap between the train and dev error.

for computer vision tasks you should thing of the human-level error as a proxy to the bayes optimal error.

- What is "Aviodable bias"? the difference between the bayes error (or the proxy) and the training error

### Understanding human-level performance

![what_to_define](2021-02-09-15-57-32.png)

- What is "humen-level" error in a case where different human groups provides different results (for example inference medical image by typical human vs. doctor vs. group of expert doctors)?
  It depends on the purpose.
  To proxy Bayes optimal error proxy hence always choose the best results.
  For other purpose such publishing a research paper or deploying a system you might choose less rigouros group

![error_example](2021-02-09-16-02-36.png)

- When your train/dev results are far away from the Bayes proxy it easy to understand where to focus on to improve, but when results approach to the human level proxy you should be carefull on how you choose where to focus on.

### Suprassing human-level performance

- When your train/dev error is better than human-level error it is hard to estimate where you sohuld focus, because you don't have an estimation the Bayes optimal error.

![surpass_ml](2021-02-09-16-10-20.png)

- What distinguish the example problems where ML surpasses human-level performance?
  - Structed data
  - Not natural perception (where human are very good)
  - Hugh amount of data (much more than human can remmember)

- Today there are also speach recognition, some image tasks, and medical application (ECG, ...) where ML surpasses human-level too

### Improving your model performance

![assumptions](2021-02-09-16-14-59.png)

- What are the two fundamental assumption of suprevised learning?

![reducing_bias_variance](2021-02-09-16-17-45.png)

- What can be done to improve aviodable bias?
- What can be done to improve variance?

## Week 2

## Error Analysis

### Carrying out error analysis

![error_analysis](2021-02-10-11-59-50.png)

Given a proposed improvement, what are the steps in error analysis?

- Get N mislabeled dev examples
- Count how many errors will be solved from the proposed imrpovement
- Calculate the precentage and decide... sometimes called "ceiling"

![multi_error_analysis](2021-02-10-12-03-29.png)

### Cleaning up incorrectly labeled data

![incorrect_train_label](2021-02-10-12-09-25.png)

- An exmaple of systematic train error that doesn't robust to DL algo: mislabeled white dogs as a cat.

![dev_test_error](2021-02-10-12-13-21.png)

- For dev/test error we can use the error analysis method and add another column to incorrectly labeled in order to estimate the porecentage of errors due mislabeling

- What are the guidelines when correcting mislabeled dev/test labeles?

![correcting_guildelines](2021-02-10-12-17-18.png)

- correcting algo got right is somethimes hard (less precentage and hard to detect)

### Build your first system quickly, then iterate

![example](2021-02-10-12-21-09.png)

- Guideline: Build your first system quickly, then iterate

## Mismatched training and dev/test set

### Training and testing on different distribution

![example_1](2021-02-10-12-29-07.png)

- When you have large dataset of different distibution from your test set and small dataset from your test set, shuffeling those datasets and split to train/test is not a good idea, because your test distribution is not similiar to your real product distribution.
- A better solution is to split your real target data set and add part of it to your train set, and use the other part for dev/test (which now come only from yout target distribution)

![example_2](2021-02-10-12-31-59.png)

### Bias and Variance with mismatched data distribution

![bias_variance](2021-02-10-12-41-40.png)

- When dev set coming from different resolution you can no longer decide naively if you have a variance problem
- What can you do in order to find if you have a variance problem when dev set coming from different distribution?
  You can define new Training-dev set, which coming from the same distribution of your training data. and analysis variance/bias over this training-dev set results.
- What is it a "data mismatch" problem? when your algo got well on the train-def set but bad on the dev set that come from a different distribution.

![bias_variance_2](2021-02-10-12-44-57.png)

- Sometimes if your dev/test set is much easier than your training set you can get dev/test better results than your training/training-dev results.

![formal](2021-02-10-12-50-28.png)

### Addressing data mismatch

![addressing](2021-02-10-14-31-17.png)

![example](2021-02-10-14-38-49.png)

- Artificial data synthesis example: adding car noise to a clean sentence audio
- Needs to be carefull and not cause to overfitting due the synthesis. for example just one hour of car noise that synthesized to all the samples might not work as expected.

![exampl_2](2021-02-10-14-42-04.png)

- Another example: trying to learn car recognition from games might looks reasonable, but typical game contains few tens of models, and the model will defently overfit for those shapes.

## Learning from multiple tasks

### Transfer learning

![transfer](2021-02-10-11-13-13.png)

- How many layers to retrain on transfer learning task of small/big data set ?
  Rule of thumb is to train only the one last output layer or maybe the last two layers, for a lot of data you can retrain all the parameters in the network.
- What is it pre-training? when retraining with all the network, we call the initial train (which trained on the first task) pre-training
- What is it fine-tuning?
  When training only few last layers as we do on small data-set we call that process fine-tuning

- Sometimes you can add more layers to the original network.
- Does it make sense to use transfer learning for new large dataset task with network learned on a small dataset? No, the original task few images are much less valuable than the target task images.

- When transfer learning make sense?
  - Task A and B have the same input x
  - You have a lot more data for task A than task B
  - Low level features from task A could be helpful for learning B

### Multi-task learning

![multi](2021-02-10-11-17-15.png)

![multi_arch](2021-02-10-11-27-45.png)

- Do you need to change the loss function?
  Needs new loss function that counts all outputs (Y is from n-dimension)
- Several outputs can be 1 (unlike softmax regression where only one output bit is 1)
- In many cases multi-task learning provides much better results than training seperate network for each task. (many common features learned together).
- Can you train multi-task learning on some partial outputs per input? (for example not mention if stio signs is there)
  Yes, you can ignore tje loss of the missing outputs. (sum only over exists labels)

- When multi-task learning makes sense?
  - Training on a set of tasks that could benefit form having shared lower-level features
  - Usually: amount of data you have for each task is quite similar (as opposed to transfer learning)
  - Can train a big enough NN to do well on all the tasks.

## End-to-end deep learning

### What is end-to-end deep learning?

#### Speech recognition example

![whatisit](2021-02-10-14-49-34.png)

- Traditional piplien: audio (x) -MFCC-> features -ML-> phonemes -> words -> transcript (y)
- DL enable end-to-end learning: audio -> transcript
- However end-to-end requires big training data (for speech recognition minimum 10K hours), for smaller data set (for example 3K hours) you might needs extra steps (such phonemes detection)

![face](2021-02-10-14-55-42.png)

- Trying to learn straight form raw image doesn't works well
- Instead, use two step approach: learn to detect face, then use the cropped zoom face as input to the identity task
- Why does it works better:
  1) Each one of the sub-problems is much simpler
  2) Have a lot of data for each of the subtasks but not so much for the whole unsplitted task 

- More examples:
  - Machine translation good for end-to-end
  - Estimating child's age needs decompose to detect bones and then classify age

### When to use end-to-end deep learning

![pros_cons](2021-02-10-15-02-11.png)

![applying](2021-02-10-15-15-00.png)

- End-to-end DL on autonomes cars is not promising for now and useually splits to several parts where DL mostly used for the object detection task.
