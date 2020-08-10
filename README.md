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
		- even now some letters are not rotated right and are not recognised accurately (sol - custom recognition model for letters)
	
        

