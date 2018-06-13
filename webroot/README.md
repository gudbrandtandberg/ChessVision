# CVweb
A Web App that uses the ChessVision project

Either run 

```python server.py 8080```

to start a server locally, or let the designated web server serve for you. 

After navigating to the ```index.html``` page, the user may select and submit an image file. 
The form is submitted using __ajax__ to a different host than serves the main web application. 

The app assumes a host is listening for __POST__ requests on __http://localhost:5000__.
The post should contain exactly one image file in the __FILES__ variable.
The response from this host is either a FEN string containing an extracted chess position, or an error message. 
