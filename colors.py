import cv2
import sys
import numpy as np 


def video(lower_bound, upper_bound):
    cap = cv2.VideoCapture(0)
    
    # Variables to store the color range based on mouse click
    selected_color = None
    
    # Mouse callback function to pick the color on click
    def pick_color(event, x, y, flags, param):
        nonlocal selected_color
        if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
            selected_color = list(frame[y, x])

    # Set the callback function for mouse events
    cv2.namedWindow('Original')
    cv2.setMouseCallback('Original', pick_color)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to HSV
        into_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # If a color has been selected, update the lower and upper bounds
        if selected_color is not None:
            # Convert the selected BGR color to HSV
            selected_color_hsv = cv2.cvtColor(np.uint8([[selected_color]]), cv2.COLOR_BGR2HSV)[0][0]
            
            lower_bound = np.array([max(0, int(selected_color_hsv[0]) - 30), 100, 100])
            upper_bound = np.array([min(255, int(selected_color_hsv[0]) + 30), 255, 255])
        else:
            lower_bound = np.array(lower_bound)
            upper_bound = np.array(upper_bound)
        
        # Create a mask based on the selected color range
        b_mask = cv2.inRange(into_hsv, lower_bound, upper_bound)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(b_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw bounding boxes around the detected contours
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Apply the mask to the original frame
        result = cv2.bitwise_and(frame, frame, mask=b_mask)
        
        # Display the frames
        cv2.imshow('Original', frame)
        cv2.imshow('Color Detector', result)
        
        # Break the loop on 'ESC' key press
        if cv2.waitKey(1) == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

# def image(path, lower_bound, upper_bound):
#     # Load the image
#     img = cv2.imread(path)
    
#     if img is None:
#         print("Error: Unable to load image.")
#         return
    
#     # Variables to store the color range based on mouse click
#     selected_color = None
    
#     # Mouse callback function to pick the color on click
#     def pick_color(event, x, y, flags, param):
#         nonlocal selected_color
#         if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
#             selected_color = list(img[y, x])

#     # Set the callback function for mouse events
#     cv2.namedWindow('Original')
#     cv2.setMouseCallback('Original', pick_color)

#     cv2.imshow('Original', img)

#     # Convert the image to HSV
#     into_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#     while selected_color is None:
#         # Break the loop on 'ESC' key press
#         if cv2.waitKey(1) == 27:
#             break
    
#     # If a color has been selected, update the lower and upper bounds
#     # Convert the selected BGR color to HSV
#     selected_color_hsv = cv2.cvtColor(np.uint8([[selected_color]]), cv2.COLOR_BGR2HSV)[0][0]
    
#     lower_bound = np.array([max(0, int(selected_color_hsv[0]) - 30), 100, 100])
#     upper_bound = np.array([min(255, int(selected_color_hsv[0]) + 30), 255, 255])

#     # Create a mask based on the selected color range
#     b_mask = cv2.inRange(into_hsv, lower_bound, upper_bound)
    
#     # Find contours in the mask
#     contours, _ = cv2.findContours(b_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Draw bounding boxes around the detected contours
#     for contour in contours:
#         if cv2.contourArea(contour) > 500:  # Filter small contours
#             x, y, w, h = cv2.boundingRect(contour)
#             cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#     # Apply the mask to the original image
#     result = cv2.bitwise_and(img, img, mask=b_mask)
    
#     while True:
#         # Display the images
#         cv2.imshow('Original', img)
#         cv2.imshow('Color Detector', result)
#         if cv2.waitKey(1) == 27:
#             break
    
#     cv2.destroyAllWindows()


def image(path, lower_bound, upper_bound):
    # Load the image
    img = cv2.imread(path)
    
    if img is None:
        print("Error: Unable to load image.")
        return
    
    # Variables to store the color range based on mouse click
    selected_color = None
    
    # Mouse callback function to pick the color on click
    def pick_color(event, x, y, flags, param):
        nonlocal selected_color, img, original_img
        if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
            selected_color = list(img[y, x])
            img = original_img.copy()


    # Set the callback function for mouse events
    cv2.namedWindow('Original')
    cv2.setMouseCallback('Original', pick_color)
    original_img = img.copy()
    
    while True:
        # Convert the image to HSV
        into_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # If a color has been selected, update the lower and upper bounds
        if selected_color is not None:
            # Convert the selected BGR color to HSV
            selected_color_hsv = cv2.cvtColor(np.uint8([[selected_color]]), cv2.COLOR_BGR2HSV)[0][0]
            
            lower_bound = np.array([max(0, int(selected_color_hsv[0]) - 20), 100, 100])
            upper_bound = np.array([min(255, int(selected_color_hsv[0]) + 20), 255, 255])
        else:
            lower_bound = np.array(lower_bound)
            upper_bound = np.array(upper_bound)
        
        # Create a mask based on the selected color range
        b_mask = cv2.inRange(into_hsv, lower_bound, upper_bound)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(b_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # # Draw bounding boxes around the detected contours
        # for contour in contours:
        #     if cv2.contourArea(contour) > 500:  # Filter small contours
        #         x, y, w, h = cv2.boundingRect(contour)
        #         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Apply the mask to the original image
        result = cv2.bitwise_and(img, img, mask=b_mask)
        
        # Display the images
        cv2.imshow('Original', img)
        cv2.imshow('Color Detector', result)
        
        # Break the loop on 'ESC' key press
        if cv2.waitKey(1) == 27:
            break


def hex_to_rgb(hex):
    r = int(hex[0:2], 16)
    g = int(hex[2:4], 16)
    b = int(hex[4:6], 16)
    return [r, g, b]


def get_lower_bound(color):
    lower_bound = []

    for c in color:
        lower_bound.append(max(0, int(c)- 50))
    
    return lower_bound

def get_upper_bound(color):
    upper_bound = []

    for c in color:
        upper_bound.append(min(255, int(c) + 50))
    
    return upper_bound


if __name__ == "__main__":

    if len(sys.argv) >= 2:
        color = hex_to_rgb(sys.argv[1])
        lower_bound = get_lower_bound(color)
        upper_bound = get_upper_bound(color)
    else:
        lower_bound = [0,0,0]
        upper_bound = [255,255,255]

    # video(lower_bound, upper_bound)
    # image('./images/image1.png', lower_bound, upper_bound)
    image('./images/image2.png', lower_bound, upper_bound)

