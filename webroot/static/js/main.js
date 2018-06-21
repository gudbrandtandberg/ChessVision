var init = function() {
    
    // The displayed chessboard (chess.js)
    var board = ChessBoard('board', 'start');

    var endpoint = document.getElementById("endpoint").innerHTML
    
    var cv_algo_url = endpoint + "cv_algo/"
    var feedback_url = endpoint + "feedback/"

    $("#upload-form").submit(function(event) {
        event.preventDefault()
        var file = document.getElementById("image-input").files[0]
        // TODO input check. 
        var formData = new FormData(this);
        
        $.ajax({
            url: cv_algo_url,
            method: "POST", 
            data: formData,
            cache: false,
            contentType: false,
            processData: false,
            success: function(data) {
                //parse data = {FEN: "...", id: "..."}
                res = JSON.parse(data)
                
                if (res.error == "false") {
                    setFEN(res.FEN)
                    document.getElementById("raw-id-input").value = res.id
                    document.getElementById("feedback-pane").style.display = "block"
                } else {
                    console.log(res)
                }
            },
            error: function(xmlHttpRequest, textStatus, errorThrown) {
                if (xmlHttpRequest.readyState == 0 || xmlHttpRequest.status == 0) {
                    alert("Connection to ChessVision server failed. It is probably sleeping..")
                    return
                } else {
                    alert(textStatus)
                }
            }
        })
    })

    $("#feedback-form").submit(function(event) {
    
        event.preventDefault()
        var formData = new FormData(this);
    
        $.ajax({
            url: feedback_url,
            method: "POST",
            data: formData, 
            cache: false,
            contentType: false,
            processData: false,
            success: function(data) {
                res = JSON.parse(data)
                document.getElementById("feedback-pane").style.display = "none"
                if (res.success == "true") {
                    alert("Thanks for your feedback!")
                } else {
                    alert("Your feedback was not taken into consideration")
                    }           
                },
            error: function(data) {
                alert(data)
                console.log(data)
                }
            })
        })
    document.getElementById('image-input').onchange = function (evt) {
        var tgt = evt.target || window.event.srcElement,
            files = tgt.files;

        // FileReader support
        if (FileReader && files && files.length) {
            var fr = new FileReader();
            fr.onload = function(data) {
                var imageData = data.target.result
                // Show preview of uploaded image
                document.getElementById("input-preview").src = imageData
            }
            fr.readAsDataURL(files[0]);
        }
        else {
            alert("Your browser must implement file uploading.")
        }
    }

    }; // end init()

var setFEN = function(fen) {
    var board = ChessBoard('board', fen);
}

$(document).ready(init);