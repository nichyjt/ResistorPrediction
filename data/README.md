# Data directory information

## raw_manual
Put images that need to be self-cropped and hacked around with here.

## raw
Put YOLO-simulated cropped images of the resistors here. 

**Naming Convention**  
Resistor naming convention shall be:
`"color-color-...-color-{number}.jpg"`  
Where the number should a unique identifier.

## labelling
Put preprocessed (data cleaned) data to be labelled in here. Open labelstudio and annotate the images.  

Once done, export the findings as `json-min` and dump the json labels into `labels_json`.

This should be done automatically by `preprocess.py`

## training
Run `labelparser` to process `labels_json` into a csv file with the important attributes hoisted out.  