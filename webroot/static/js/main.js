var init = function() {
            
    var board = ChessBoard('board', 'start');
    
    $("form#theform").submit(function(event) {
        event.preventDefault()
        var file = document.getElementById("imageinput").files[0]
        // TODO input check. 
        var formData = new FormData(this);
        
        $.ajax({
            url: "http://localhost:7777/",
            method: "POST", 
            data: formData,
            success: function (data) {
                alert(data)
            },
            cache: false,
            contentType: false,
            processData: false,
            success: function(data) {
                setFEN(data)
            },
            error: function(xmlHttpRequest, textStatus, errorThrown) {
                if(xmlHttpRequest.readyState == 0 || xmlHttpRequest.status == 0) {
                    alert("quitting")
                    return
                } else {
                    alert(textStatus)
                }
            }
        })

        var reader = new FileReader();
        reader.onload = function(data) {
            var imageData = data.target.result
            // Show preview of uploaded image
            document.getElementById("inputpreview").src = imageData
            // Send base64 encoded image to backend..
        }
        reader.readAsDataURL(file);
})
}; // end init()
var setFEN = function(fen) {
    var board = ChessBoard('board', fen);
}
    
$(document).ready(init);