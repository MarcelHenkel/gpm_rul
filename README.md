# Generation and prediction models for timeseries sequences 
This framwork consits of generative and predictive models for remaining useful life estimation. This framework consits three prediction and two generative adverserial newtorks (GANs). For the three prediction models use following machine learning models:
  1. Recurrent Neural Network
  2. Transformer Network
  3. Random Forest.
As an input those models take a timeseries sequence of defined length. A timeseries sequence is a 2D window of measurements, where each row corresponds to a timestamp and each column corresponds to a measurement. 

This framework can easily be adapted to different datasets. Testing was performed with three popular datasets, which are provided by NASA under the following url:
bearing dataset (4.), Li-Ion battery dataset (5.) and areo turbine dataset (6.), which can be downloaded here: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

These Datasets be downloaded from the url below and can be unpacked in the appropriate directories to run the code. The generative Model uses the 2014 published GAN structure. 

Two variants of this model are used here. 
  1. CRGAN, which is based on Paper "time-series regeneration with convolutional recurrent generative adversarial network for remaining useful
life estimation" (https://www.semanticscholar.org/paper/Time-Series-Regeneration-With-Convolutional-Network-Zhang-Qin/55dd7715123a5ada27ad2b8db996a54170dcf9bd)
  2. Stacked GAN, which is based on Paper "MTSS-GAN: Multivariate Time Series Simulation Generative Adversarial Networks" (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3616557) and the code which https://github.com/firmai/mtss-gan.

All models are implemented using the keras API of tensorflow. 
