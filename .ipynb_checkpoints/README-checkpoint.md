# ✋🏼🛑 One-Stop-for-COVID-19-Infection-and-Lung-Segmentation-plus-Classification

The project is a complete COVID-19 detection package comprising of 3 tasks:<br />
<br />
➤ **Task 3:** Lung Segmentation<br />
➤ **Task 1:** COVID-19 Infection Segmentation<br />
➤ **Task 2:** COVID-19 Classification<br />
 
<table border="0">
 <tr>
    <td><b style="font-size:30px">Sample data</b></td>
 </tr>
 <tr>
    <td><img src="pics/sample_data.png" width="1200" height="275" /></td>
 </tr>
</table>


#### -----------------------------------------------------------------------------------------------------------------------------------------

### ➥ <ins> Some Instructions and Guidelines for Code Execution </ins>
1. All scripts are exactly the same with the notebooks having same titles.
2. Any necessary package to be installed is placed on top of each of the six scripts/notebooks.



#### -----------------------------------------------------------------------------------------------------------------------------------------


### ➥ <ins> Following are the details about the scripts/notebooks </ins>

★ **task1_preprocessing_plus_unet_with_comments.py** ---> Contains the maximum number of comments and explanation. Any doubt, if persists probably could be rectified here. It contains the UNet training for infection mask prediction (Task- 1)



#### -----------------------------------------------------------------------------------------------------------------------------------------


*Note: Not all are displayed here, rest could be fetched after running notebooks*

```
RESULT TABLE
+----------------------------------------+----------------------------+-----------------------+---------------------------+------------------------+--------+--------------+----------+
|                                        |    Dice (Mean of folds)    |  IOU (Mean of folds)  | Precision (Mean of folds) | Recall (Mean of folds) | AUCROC |   F1 Score   | Accuracy |
+----------------------------------------+----------------------------+-----------------------+---------------------------+------------------------+--------+--------------+----------+
| Task1: Infection Segmentation (3-fold) |            0.948           |         0.903         |           0.947           |         0.950          |    -   | same as dice |     -    |
+----------------------------------------+----------------------------+-----------------------+---------------------------+------------------------+--------+--------------+----------+
| Task1: Infection Segmentation (4-fold) |            0.956           |         0.917         |           0.955           |         0.958          |    -   | same as dice |     -    |
+----------------------------------------+----------------------------+-----------------------+---------------------------+------------------------+--------+--------------+----------+
|          Task2: Classification         |              -             |           -           |           0.987           |          0.989         |  0.998 |     0.988    |   0.982  |
+----------------------------------------+----------------------------+-----------------------+---------------------------+------------------------+--------+--------------+----------+
|        Task3: Lung Segmentation        |            0.984           |         0.969         |             -             |            -           |    -   | same as dice |     -    |
+----------------------------------------+----------------------------+-----------------------+---------------------------+------------------------+--------+--------------+----------+
|                                        | Note: precision and recall | values are as per the | best threshold for dice   |                        |        |              |          |
+----------------------------------------+----------------------------+-----------------------+---------------------------+------------------------+--------+--------------+----------+
```
#### -----------------------------------------------------------------------------------------------------------------------------------------
# Preprocessing Stage

1.) Removing incomplete and fauty images <br />
2.) Separate model for empty mask prediction <br />
3.) Use of Contrast Limited Adaptive Histogram Equalization (
) for image enhancement <br />
4.) Cropping the Region of Interst (ROI) using Otsu's binarization and other approaches <br />
5.) Data Augmentation <br />


<table border="0">
 <tr>
    <td><b style="font-size:30px">Cropping out contour with the largest area</b></td>
 </tr>
 <tr>
    <td><img src="pics/cropper.JPG" width="1000" height="300" /></td>
 </tr>
</table>


<table border="0">
 <tr>
    <td><b style="font-size:30px">CLAHE only Output</b></td>
    <td><b style="font-size:30px">Before-After Image (after all preprocessing steps)</b></td>
 </tr>
 <tr>
    <td><img src="pics/CLAHE.JPG" width="500" height="500" /></td>
    <td><img src="pics/before_after_preprocessing.PNG" width="500" height="300" /></td>
 </tr>
</table>


<table border="0">
 <tr>
    <td><b style="font-size:30px">Augmentated CT Scans (Task: 1)</b></td>
 </tr>
 <tr>
    <td><img src="pics/new_aug_cts.png" widthcts_aug_again.png="1000" height="150" /></td>
 </tr>
</table>

<table border="0">
 <tr>
    <td><b style="font-size:30px">Augmented Infections Masks (Task: 1)</b></td>
 </tr>
 <tr>
    <td><img src="pics/new_aug_infections.png" width="1000" height="150" /></td>
 </tr>
</table>


<table border="0">
 <tr>
    <td><b style="font-size:30px">Augmentated CT Scans (Task: 3)</b></td>
 </tr>
 <tr>
    <td><img src="pics/cts_aug_again.png" widthcts_aug_again.png="1000" height="150" /></td>
 </tr>
</table>

<table border="0">
 <tr>
    <td><b style="font-size:30px">Augmented Lung Masks (Task: 3)</b></td>
 </tr>
 <tr>
    <td><img src="pics/lungs_aug_again.png" width="1000" height="150" /></td>
 </tr>
</table>


<table border="0">
 <tr>
    <td><b style="font-size:30px">DICE v/s IOU and some relationship </b></td>
    <td><b style="font-size:30px"></b></td>
 </tr>
 <tr>
    <td>• Dice (S) = 2|𝐴∩𝐵|)/(|𝐴|  + |𝐵|) = 2𝑇𝑃/(2𝑇𝑃+𝐹𝑃+𝐹𝑁) <br/> 
	• IOU (J) = (|𝐴∩𝐵|)/(|𝐴∪𝐵|) = 𝑇𝑃/(𝑇𝑃+𝐹𝑃+𝐹𝑁) <br/>
	• J = S/(S-2) <br/>
	• 𝜕𝐽/𝜕𝑆 = 2/[(2 −𝑠)]^2  <br/>
	    <br/>
	    • <b>Interpretations:</b> <br/>
	<br/>
	    • We can say that IOU(J) is always more punishing or has less value than the corresponding dice(S) at the same threshold. <br/> <br/>
	    • Another takeaway is that 𝜕𝐽/𝜕𝑆 = 2/[(2 −𝑠)]^2 which is basically the slope and is a continuous 
    increasing function for  S ∈ [0,1]. <br/> <br/>
	    • For (S = 0.586, J = 0.414), the value of slope equals to 1. <br/> <br/>
	    • Above two points combinedly establishes a relationship that for all S > 0.586, 
    rate of increase of IOU is greater than the dice whereas for all S < 0.586, the rate of increase 
    of dice is greater than the IOU. <br/> <br/>
	</td>
    <td><img src="pics/vennd.PNG" width="750" height="300" /></td>
 </tr>
</table>



<table border="0">
 <tr>
    <td><b style="font-size:30px"></b></td>
    <td><b style="font-size:30px"></b></td>
 </tr>
 <tr>
    <td><img src="pics/dics vs iou.PNG" width="500" height="400" /></td>
    <td><img src="pics/delj by s.PNG" width="500" height="400" /></td>
 </tr>
</table>


# Training Stage 
(Remains same for all tasks)

<table border="0">
 <tr>
    <td><b style="font-size:30px">Exponential decaying LR (Step and continuous)</b></td>
    <td><b style="font-size:30px">Exponential decaying LR (Step and continuous) with Variation</b></td>
 </tr>
 <tr>
    <td><img src="pics/exponential lrs.JPG" width="500" height="300" /></td>
    <td><img src="pics/expo LR variations.PNG" width="500" height="300" /></td>
 </tr>
</table>

<table border="0">
 <tr>
    <td><b style="font-size:30px">Cosine Annealing Cyclical LR</b></td>
    <td><b style="font-size:30px">Blend of Cosine Annealing and Exponential Decay</b></td>
 </tr>
 <tr>
    <td><img src="pics/cosine_only.PNG" width="500" height="300" /></td>
    <td><img src="pics/cosine_expo.PNG" width="500" height="300" /></td>
 </tr>
</table>

# Results with UNet (Task: 1)

<table border="0">
 <tr>
    <td><b style="font-size:30px">Trianing curve for Dice Coefficient</b></td>
    <td><b style="font-size:30px">Training curve for BCE + Dice Loss</b></td>
 </tr>
 <tr>
    <td><img src="pics/Best training curve_dice.png" width="400" height="300" /></td>
    <td><img src="pics/Best training curve_loss.png" width="400" height="300" /></td>
 </tr>
</table>

<table border="0">
 <tr>
    <td><b style="font-size:30px">Optimizing threshold with small step size</b></td>
 </tr>
 <tr>
    <td><img src="pics/Infection mask_Optimizing threshold more in best threshold segment.JPG" width="1000" height="300" /></td>
 </tr>
</table>


<table border="0">
 <tr>
    <td><b style="font-size:30px">Precision and recall curves v/s thresholds</b></td>
 </tr>
 <tr>
    <td><img src="pics/Infections mask_Precision and Recall.JPG" width="550" height="400" /></td>
 </tr>
</table>


<table border="0">
 <tr>
    <td><b style="font-size:30px">Some Actual Vs Predicted Masks</b></td>
    <td><b style="font-size:30px">Some Actual Vs Predicted Masks</b></td>
 </tr>
 <tr>
    <td><img src="pics/new infection predictions1.PNG" width="500" height="500" /></td>
    <td><img src="pics/new infection predictions2.PNG" width="500" height="500" /></td>
 </tr>
</table>

# 4-Fold Cross-validation Results on Task: 1 (UNet)

<table border="0">
 <tr>
    <td><b style="font-size:30px">4-fold threshold vs split number dataframe for DICE</b></td>
    <td><b style="font-size:30px">Brief report acquired from dataframe</b></td>
 </tr>
 <tr>
    <td><img src="pics/4 dices df.PNG" width="300" height="300" /></td>
    <td><img src="pics/4 dice report.PNG" width="700" height="50" /></td>
 </tr>
</table>

<table border="0">
 <tr>
    <td><b style="font-size:30px">4-fold threshold vs split number dataframe for IOU</b></td>
    <td><b style="font-size:30px">Brief report acquired from dataframe</b></td>
 </tr>
 <tr>
    <td><img src="pics/4 iou df.PNG" width="300" height="300" /></td>
    <td><img src="pics/4 iou report.PNG" width="700" height="50" /></td>
 </tr>
</table>

<table border="0">
 <tr>
    <td><b style="font-size:30px">4-fold threshold vs split number dataframe for PRECISION</b></td>
    <td><b style="font-size:30px">Brief report acquired from dataframe</b></td>
 </tr>
 <tr>
    <td><img src="pics/4 precision df.PNG" width="300" height="300" /></td>
    <td><img src="pics/4 precision report.PNG" width="700" height="50" /></td>
 </tr>
</table>

<table border="0">
 <tr>
    <td><b style="font-size:30px">4-fold threshold vs split number dataframe for RECALL</b></td>
    <td><b style="font-size:30px">Brief report acquired from dataframe</b></td>
 </tr>
 <tr>
    <td><img src="pics/4 recall df.PNG" width="300" height="300" /></td>
    <td><img src="pics/4 recall report.PNG" width="700" height="50" /></td>
 </tr>
</table>


<table border="0">
 <tr>
    <td><b style="font-size:30px">Some Actual v/s Predicted Masks by 4 Unet models of 4-fold Cross-Validation</b></td>
 </tr>
 <tr>
    <td><img src="pics/4fold predicted vs actual.PNG" width="900" height="600" /></td>
 </tr>
</table>


# Results with UNet++ (Task: 1)

<table border="0">
 <tr>
    <td><b style="font-size:30px">Training curve for Dice Coefficient</b></td>
    <td><b style="font-size:30px">Training curve for BCE + Dice Loss</b></td>
 </tr>
 <tr>
    <td><img src="pics/training dice unetpp.png" width="350" height="300" /></td>
    <td><img src="pics/training loss unetpp.png" width="350" height="300" /></td>
 </tr>
</table>


<table border="0">
 <tr>
    <td><b style="font-size:30px">Optimizing threshold with small step size</b></td>
 </tr>
 <tr>
    <td><img src="pics/threshold unetpp2.PNG" width="800" height="300" /></td>
 </tr>
</table>


# Results with CNN (Task: 2)

<table border="0">
 <tr>
    <td><b style="font-size:30px">Classification loss curve</b></td>
 </tr>
 <tr>
    <td><img src="pics/loss curve classification.png" width="600" height="380" /></td>
 </tr>
</table>


<table border="0">
 <tr>
    <td><b style="font-size:30px">Distribution of TN, TP, FN, FP with Threshold 0.50</b></td>
    <td><b style="font-size:30px">Distribution of TN, TP, FN, FP with Best Threshold 0.81</b></td>
 </tr>
 <tr>
    <td><img src="pics/distribution 0.5.PNG" width="500" height="400" /></td>
    <td><img src="pics/distribution 0.81.PNG" width="500" height="400" /></td>
 </tr>
</table>


<table border="0">
 <tr>
    <td><b style="font-size:30px">ROC Curve with Threshold 0.50</b></td>
    <td><b style="font-size:30px">ROC Curve with Threshold 0.81</b></td>
 </tr>
 <tr>
    <td><img src="pics/roc 0.5.PNG" width="500" height="500" /></td>
    <td><img src="pics/roc 0.81.PNG" width="500" height="500" /></td>
 </tr>
</table>

<table border="0">
 <tr>
    <td><b style="font-size:30px">Confusion matrix with Threshold 0.50</b></td>
    <td><b style="font-size:30px">Confusion matrix with Best Threshold 0.81</b></td>
 </tr>
 <tr>
    <td><img src="pics/cm 0.5.PNG" width="500" height="550" /></td>
    <td><img src="pics/cm 0.81.PNG" width="500" height="550" /></td>
 </tr>
</table>


# Results with UNet (Task: 3)

<table border="0">
 <tr>
    <td><b style="font-size:30px">Training curve for Dice Coefficient</b></td>
    <td><b style="font-size:30px">Training curve for BCE + Dice Loss</b></td>
 </tr>
 <tr>
    <td><img src="pics/Lung mask training dice curve.jpg" width="350" height="300" /></td>
    <td><img src="pics/Lung mask training loss curve.jpg" width="350" height="300" /></td>
 </tr>
</table>


<table border="0">
 <tr>
    <td><b style="font-size:30px">Optimizing threshold with small step size</b></td>
 </tr>
 <tr>
    <td><img src="pics/threshold 2 lung seg.PNG" width="1000" height="300" /></td>
 </tr>
</table>


<table border="0">
 <tr>
    <td><b style="font-size:30px">Actual v/s Predicted Lung Masks</b></td>
    <td><b style="font-size:30px">Actual v/s Predicted Lung Masks</b></td>
 </tr>
 <tr>
    <td><img src="pics/predicted vs actual lung seg.PNG" width="500" height="500" /></td>
    <td><img src="pics/predicted vs actual lung seg2.PNG" width="500" height="500" /></td>
 </tr>
</table>

`
`
`
`
`
` 
`
`
# License
MIT License

Copyright (c) 2020 Rohit Verma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
