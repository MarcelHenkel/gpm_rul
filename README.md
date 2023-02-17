# gpm_rul
This framwork consits of generative and predictive models for remaining useful life estimation. This framework consits three prediction and two generative adverserial newtorks (GANs). For the three prediction models use following machine learning models:
  1. Recurrent Neural Network
  2. Transformer Network
  3. Random Forest.
As an input those models take a timeseries sequence of defined length. A timeseries sequence is a 2D window of measurements, where each row corresponds to a timestamp and each column corresponds to a measurement. 

This framework can easily be adapted to different datasets. The framework is tested on three popular datasets, which are provided by NASA under the following links:
  1. Areo turbine dataset (cmapss), 
  2. Ball bearing dataset,
  3. Lithium battery dataset,

All models are implemented using the keras API of tensorflow. 
