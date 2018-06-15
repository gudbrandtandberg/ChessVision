var init = function() {
        
    var board = ChessBoard('board', 'start');

    var cv_algo_url = "http://40.113.67.136:8080/cv_algo/"
    var feedback_url = "http://40.113.67.136:8080/feedback/"
    
    $("form#theform").submit(function(event) {
        event.preventDefault()
        var file = document.getElementById("imageinput").files[0]
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
                    //$("#feedback_pane").css("visibility", "visible")
                    document.getElementById("raw-id-input").value = res.id
                    //document.getElementById("feedback_pane").style.visibility = "visible"
                } else {
                    alert(res.errMsg)
                }
            },
            error: function(xmlHttpRequest, textStatus, errorThrown) {
                if(xmlHttpRequest.readyState == 0 || xmlHttpRequest.status == 0) {
                    alert("prematurely aborting ajax request..")
                    return
                } else {
                    alert(textStatus)
                }
            }
        })
    })

    $("form#feedback_form").submit(function(event) {
    
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
                console.log(res)
                if (res.success == "true") {
                    alert("Thanks for your feedback!")
                } else {
                    alert("Your feedback was not taken into consideration")
                    }              
                },
            error: function(data) {
                alert(data)
                }
            })
        })
    document.getElementById('imageinput').onchange = function (evt) {
        var tgt = evt.target || window.event.srcElement,
            files = tgt.files;

        // FileReader support
        if (FileReader && files && files.length) {
            var fr = new FileReader();
            fr.onload = function(data) {
                var imageData = data.target.result
                // Show preview of uploaded image
                document.getElementById("inputpreview").src = imageData
            }
            fr.readAsDataURL(files[0]);
        }

        // Not supported
        else {
            // fallback -- perhaps submit the input to an iframe and temporarily store
            // them on the server until the user's session ends.
        }
    }

    }; // end init()

var setFEN = function(fen) {
    var board = ChessBoard('board', fen);
}

$(document).ready(init);