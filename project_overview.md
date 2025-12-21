See https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge/overview


## Description 
Join the hunt for stars being torn apart by black holes!

The 10-Year Legacy Survey of Space and Time (LSST) using the Vera C. Rubin Observatory will soon begin. It is expected to revolutionise time-domain astronomy, discovering 100x more astronomical transients (such as supernovae) than we’ve ever discovered.

With this wealth of new data will come an inherent challenge, as we do not have the available resources to classify all of the objects that we discover. In astronomy, classifications are generally performed by using a spectroscopic instrument on a telescope to get a spectra from an object, showing you the intensity of the light with respect to wavelength and allowing you to identify emission or absorption features that tell you more about the object you’re observing. LSST will require us to be able to identify which objects are likely interesting based purely on their lightcurves (change in brightness over time) in order to choose which to devote our limited follow-up resources to.

In particular, we’re interested in a class of objects called tidal disruption events (TDEs). These occur whenever an unlucky star is ripped apart by the immense gravitational forces it experiences as it approaches too close to a supermassive black hole. TDEs are a relatively recent astronomical discovery with a very small catalogue of observed objects (~100). However, those that we have found have proven to be tremendously scientifically valuable, particularly for investigating the properties and feeding conditions of otherwise very difficult to observe black holes. Our research capabilities are currently limited by our small sample size, but LSST provides the opportunity to solve that issue – provided we are able to make the best use of it. For more information on tidal disruption events see https://arxiv.org/abs/2104.14580 and https://arxiv.org/abs/2008.05461.

To help meet this challenge, we introduce the Many Artificial LSST Lightcurves based on Observations of Real Nuclear transients (MALLORN) Classifier Challenge. This competition invites participants to help us prepare for LSST by developing machine learning algorithms capable of photometrically identifying TDEs within a simulated LSST dataset. The light curves in this challenge are simulations based on real observations from the Zwicky Transient Facility (ZTF) (see https://www.ztf.caltech.edu).

Google Colab notebooks are available describing how to use the data (notebooks/mallorn_using_data.py) and the methods used to generate the data (notebooks/mallorn_production.py).

The paper describing the production of the data set and the structure of the data challenge is now submitted to arXiv (see https://arxiv.org/abs/2512.04946) and has been submitted for publication

Acknowledgements

DM acknowledges a studentship funded through the Leverhulme Interdisciplinary Network on Algorithmic Solutions. MN is supported by the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No.~948381).

The data used to produce this data set was sourced from the Zwicky Transient Facility. The SNCosmo package and the models included within it were used to generate data for other bands from the existing ZTF observations. The Rubin Survey Simulator baseline was used to implement accurate LSST cadence into the lightcurves


## Evaluation

### Evaluation Metric

Submissions will be evaluated using the F1 score, with the primary objective of correctly identifying TDEs within the dataset. While participants may choose to classify other types of transients, only the accuracy of TDE identification will contribute to the final evaluation. The final submission should therefore contain a binary prediction variable, indicating whether each source is classified as a TDE (1) or not (0).

The F1 score is defined as:

$F1 = \frac{2 \times precision \times recall}{precision+recall}$


where $precision = TP / (TP+FP)$ and $recall = TP/(TP+FN)$

Here, TP, FP, and FN represent the number of true positives, false positives, and false negatives, respectively.

The F1 score is preferred over simple accuracy in this task because the dataset is highly imbalanced, with TDEs being significantly rarer than other classes. This metric provides a balanced measure of performance, rewarding models that achieve a good trade-off between recall (detecting as many true TDEs as possible) and precision (avoiding excessive false positives).

Submission File

The Submission file should contain the predicted class (0 or 1) for each sources in the test_log.csv. The file should be a csv file, with a header in the following format.

```
object_id,prediction
Eluwaith_Mithrim_nothrim,0
Eru_heledir_archam,0
Gonhir_anann_fuin,0
Gwathuirim_haradrim_tegilbor,0
achas_minai_maen,0
adab_fae_gath,0
adel_draug_gaur,0
......
```

## Dataset description

File Descriptions

Split [01-20] : The repository for that respective split of the MALLORN data set. Each are roughly equal in size.
[train/test]_full_lightcurves.csv : Time series observations of objects in each respective filter. Multiple objects are saved to the one csv file, and can be extracted using the object_id. Contains the full lightcurves for the objects.
[train/test]_log.csv : Additional information about the objects such as the extinction and redshift of the objects. This also includes the SpecType(true classification of the object) and target (binary variable for TDE and non-TDE). Further the split column in the log files can be used to identify the respective split folder for a given object_id.
sample_submission.csv : A valid submisison file.

The notebook/mallorn_using_data.py  notebook provides an overview of loading in the lightcurve for a chosen object.

Log File Column Descriptions

object_id : A combination of 3 Sindarin (Elvish) words which provides a unique identifier for each object.
Z : The redshift of the object. For the training data these are spectroscopic redshift values, which have negligible error. For the testing set these are simulated photometric redshift values which do have a corresponding error.
Z_err : The error in the redshift. This value is left blank for the training data for the reasons described above.
EBV : The extinction coefficient (E(B-V)) value. This provides a measure for the amount the light of the source has been obscured by dust on its path to reach us. This value can be used to de-extinct the flux measurements through a method shown in the ‘Using_the_Data’ notebook.
SpecType : The spectroscopically defined type of the object. This value is given in the training data to allow for effective training of classifiers. However, it is left blank for the testing data.
English Translation : An English translation of the Object_ID. Unimportant for the classifier challenge but a fun inclusion.
split : The name of the split folder that contains the the full lightcurve file for a given object.
target (only for training set): The target variable for TDE classification (derived from SpecType). The value is 1 if the source is a TDE and 0 if anything else.

Data File Column Descriptions

object_id : Same as described above.
Time (MJD) : The date of the observation. In units of MJD (Modified Julian Date) which are commonly used in astronomy. The number indicates the number of days since 17/11/1858.
Flux : The measured amount of light from the observation. Given in units of microjansky (μJy). These values are unextincted and need to be corrected for that using the ‘EBV’ value for the corresponding object from the relevant log file.
Flux_err : The uncertainty in the flux measurement described above.
Filter : The observation filter used for that respective observation. LSST will use six different filters: u, g, r, i, z & y. The filter corresponds to a particular wavelength range. The time series data for each filter should be regarded separately as they can behave differently for a given type of object.

Additional Information

Gaps in observations: Due to the nature of LSST’s cadence there will be gaps between observations for a respective band. The frequency of observations are also effected by the position of the Sun and simulated weather – both of which can result in larger gaps between observations. As a result, the overall quality of the lightcurves can vary. However, measures have been taken to ensure the overall quality of the lightcurves included within this classification challenge.

Nuclear extragalactic objects: All of the simulated transients included in this data set were produced from real observations of nuclear transients by ZTF. Therefore all of the simulated objects are regarded as nuclear transients. There is no additional contextual information provided with the position of the transient. There are no galactic transients included within the MALLORN data set as it is assumed that astronomers will be able to filter out those objects.

Types included: The following types of extragalactic transient are included in the MALLORN data set: SN Ia, SN Ia-91T-like, SN Ia-91bg-like, SN Ia02cx-like, SN Ia-pec, SN Ib, SN Ib/c, SN Ic, SN Ic-BL, SN II, SN IIb, SN IIn, SLSN-I, SLSN-II, TDEs & AGN.

Negative flux measurements: All of the flux measurements are taken with respect to a baseline value for that sky position. This can therefore result in negative flux values if the flux varies below that baseline value. These are particularly prevalent for AGN.