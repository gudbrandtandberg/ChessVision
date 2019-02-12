# ChessVision Wiki!

## Chessvision Architecture

The ChessVision algorithm is exposed on a Amazon Lambda function.
The lambda function can be invoked from many languages using the AWS SDK. 
The format for request data is a serialized  object:

```
data = {"image": "<image base 64>", "flip": "<true or false>"}
```

The server responds with a object like this

```
result = { "statusCode": <number>, "body": <application/json encoded result> }
```

## Call ChessVision from javascript

```javascript
// Initialize the Amazon Cognito credentials provider
AWS.config.region = 'eu-central-1'; // Region
AWS.config.credentials = new AWS.CognitoIdentityCredentials({
    IdentityPoolId: 'eu-central-1:d06d1df9-443e-49e3-84e8-d90aacb9b333',
});
    
var lambda = new AWS.Lambda({region: "eu-central-1"});
lambda.config.credentials = AWS.config.credentials;
lambda.config.region = AWS.config.region;

// Get the base64 encoded version of the input image. 
// Make sure it is 512x512 and only contains the data part of the b64 url.
var b64image = cropper.getCroppedCanvas({width: 512, height: 512}).toDataURL('image/jpeg', 0.9).split(",")[1];

// The payload format 
var payload = JSON.stringify({image: b64image, flip: "false"});

var params = {
    Payload: payload,
    FunctionName : "arn:aws:lambda:eu-central-1:580857158266:function:chessvisionClient",
    InvocationType : "RequestResponse"
    };
    
lambda.invoke(params, function(error, data) {
    if (error) {
        prompt(error);
    } else {
        var payload = JSON.parse(data.Payload);
        var body = JSON.parse(payload.body);
        // do something with body, it has the format:
        // body = {"FEN": "<fen string>"}
    }
});
```