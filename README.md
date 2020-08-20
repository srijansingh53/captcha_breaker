# Captcha_Breaker

Decoding captcha with deep learning

# Progress

1. Cleaning data done and label.csv created
2. Initial preprocessing of images done.
    - morphological closing with an appropriate kernel.
    - otsu binarization done
    - processed images saved in drive - processed folder

    **Observations**
        - easyocr applied - detecting around 60 percent. Problem is with highly tilted letters and dots.(solution proposed-see 3)

3. MinAreaRect concept applied.
    - found contours and removed small contours(dot) by thresholding contourArea, thus having a list of good_contours
    - extracted minAreaRect after sorting good_contours
    - rotated accordingly clockwise and anti-clockwise <-45 and >-45 degrees
    
    **Observations**
        - problem with cropping and rotating piecewise letters- letters have to be cropped(done)
4. Applied easyocr and tesseract to stiched image
	- cropped according to minAreaRect
	- padded with black pixels
	- concatenated with all letters
	
	**Observations**
		- even now some letters are not rotated right and are not recognised accurately, generalization problem of IP (sol - custom recognition model for letters)
	
5. Angle rotation technique modified (after getting error)
	- 4 cases of angles handled
	- padding size need to be increased
	- data pipeline to made for training set
	
	**Observations**
		- Skewness of different type can't be generalised with IP techniques, dl model needed
		
        
6. 5 DL models created with 3/4 cnn and 1 fcn. 
	- got test scc of 97/98 % and train acc of 98/99 %
	- may be sota architectures required.
	- callbacks to be done with pytorch lightning
	
