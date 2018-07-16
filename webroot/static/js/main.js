// 
// main.js -- all the chessvision website stuff
//

// global variablaes
var board, cropper, cropperOptions;
var input, canvas, context, endpoint;

var sizeCanvas = function() {
    canvas = document.getElementById("image-preview");
    canvas.height = canvas.width;
}

// initialize variables
var init = function() {

    sizeCanvas()

    // Initialize chessboard (chessboard.js)
    board = ChessBoard( 'board', {position: "8/8/8/8/8/8/8/8 w KQkq -",
                        dropOffBoard: "trash",
                        orientation: "white",
                        sparePieces: false,
                        showErrors: "console"
                        });
    $(window).resize(board.resize);

    // Initialize cropper (cropper.js)
    cropperOptions = { dragMode: "move",
                viewMode: 3,
                cropBoxMovable: false,
                cropBoxResizable: false,
                autoCropArea: 1.0,
                aspectRatio: 1.0,
                background: true,
                guides: true,
                center: false,
                highlight: true,
                modal: true,
                toggleDragModeOnDblclick: false,
                ready: function(event) {
                    
                },
                crop: function(event) {
                    
                  }
                }
    $("#image-preview").cropper(cropperOptions);

    // Hook up event handlers
    $("#upload-form").submit(extractBoard)      
    document.getElementById("image-input").onchange = imageInputChanged

} // end init

// Fires every time a new file is selected (or file-choosing is cancelled)
var imageInputChanged = function() {

    canvas = document.getElementById("image-preview");
    context = canvas.getContext('2d');
    var reader = new FileReader();
    
    reader.addEventListener("loadend", function(arg) {

      var src_image = new Image();
      
      src_image.onload = function() {
        
        canvas.height = src_image.height;
        canvas.width = src_image.width;
        context.drawImage(src_image, 0, 0);
        
        cropper = $("#image-preview").data("cropper");
        cropper.replace(this.src)
      }

      src_image.src = this.result;
      
    });
    if (this.files) {
        reader.readAsDataURL(this.files[0]);
        $("#board-container").hide()
        $("#preview-container").show()
        $("#board-state-pane").show()
    } else {
        alert("no file chosen")
    }
};

var setSpinner = function() {
    var button = $("#submit-btn")
    button.html('Extracting... <img src="/static/img/spinner.gif" width="20px"/>')
}

var unsetSpinner = function() {
    var button = $("#submit-btn")
    button.html("Extract")
}


// Package image and metadata in a formdata object and send to server
var extractBoard = function(event) {
    event.preventDefault()

    endpoint = document.getElementById("endpoint").innerHTML
    
    var cv_algo_url = endpoint + "cv_algo/"

    // option: minWidth
    if (cropper == undefined) {
        alert("Please upload a photo first!");
        return;
    }
    dataURL = cropper.getCroppedCanvas().toDataURL('image/jpeg')
    
    setSpinner()

    var blobBin = atob(dataURL.split(',')[1]);
    var array = [];
    for (var i = 0; i < blobBin.length; i++) {
        array.push(blobBin.charCodeAt(i));
    }
    var file = new Blob([new Uint8Array(array)], {type: 'image/jpeg'});
    flip = document.getElementById("reversed-input").checked ? "true" : "false"

    var formData = new FormData();
    formData.append("file", file);
    formData.append("flip", flip);

    $.ajax({
        url: cv_algo_url,
        method: "POST", 
        data: formData,
        cache: false,
        contentType: false,
        processData: false,
        timeout: 3000,
        success: uploadSuccess,
        error: function(xmlHttpRequest, textStatus, errorThrown) {
            unsetSpinner()
            if (xmlHttpRequest.readyState == 0 || xmlHttpRequest.status == 0) {
                alert("Connection to ChessVision server failed. It is probably sleeping..")
                return
            } else {
                alert(textStatus)
            }
        }
    })
}

var uploadSuccess = function(data) {
    //parse data = {FEN: "...", id: "..."}
    
    unsetSpinner()
    res = JSON.parse(data)
    
    if (res.error == "false") {
        $("#board-container").show()
        board.resize()
        $("#preview-container").hide()
        $("#edit-analyze-pane").show()
        setFEN(res.FEN)
        document.getElementById("raw-id-input").value = res.id
        
    } else {
        console.log(res)
    }
}

$("#edit-btn").on("click", function(e) {
    
    var orientation = document.getElementById("reversed-input").checked ? "black" : "white"
    var pos = board.position("fen")
    board = ChessBoard( 'board', {
    position: pos,
    orientation: orientation,
    dropOffBoard: "trash",
    sparePieces: true,
    });
    //$(window).resize(board.resize);

    $("#feedback-pane").show()
    $("#submit-pane").hide()
    $("#edit-analyze-pane").hide()
})

$("#feedback-form").submit(function(event) {
    event.preventDefault()
    var feedback_url = endpoint + "feedback/"
    var position = board.position()
    var flip = document.getElementById("reversed-input").checked ? "true" : "false"

    var formData = new FormData(this);
    formData.append("position", JSON.stringify(position))
    formData.append("flip", flip)

    $.ajax({
        url: feedback_url,
        method: "POST",
        data: formData, 
        cache: false,
        contentType: false,
        processData: false,
        success: function(data) {
            res = JSON.parse(data)
            $("#feedback-pane").hide()
            var pos = board.position("fen")
            var orientation = document.getElementById("reversed-input").checked ? "black" : "white"
            board = ChessBoard( 'board', {
                position: pos,
                orientation: orientation,
                sparePieces: false,
                });
            $("#submit-pane").show()
            $("#edit-analyze-pane").show()

            if (res.success == "true") {
                alert("Thanks for your feedback!")
            } else {
                alert("Something went wrong, your feedback was not taken into consideration")
                }           
            },
        error: function(data) {
            alert("Feedback-ajax failed..")
            console.log(data)
            }
        })
    })

var toggleTurn = function() {
    $('input[type="radio"]').not(':checked').prop("checked", true);
};

$("#analyze-btn").on("click", function() {
    
    var analyze_url = endpoint + "analyze/"
    var formData = new FormData(this);

    // get valid fen from board + input tags.
    var fen = board.fen()
    fen = expandFen(fen)

    formData.append("FEN", fen)

    $.ajax({
        url: analyze_url,
        method: "POST",
        data: formData, 
        cache: false,
        contentType: false,
        processData: false,
        success: function(data) {
            res = JSON.parse(data)
            if (res.success == "false") {
                alert("Analysis failed..")
                return
            }
            var bestMove = res.bestMove
            if (bestMove.length == 5) {
                alert("strange move!")
            }
            src = bestMove.substr(0, 2)
            dst = bestMove.substr(2, 2)
            move = src.concat("-", dst)
            board.move(move)
            toggleTurn()

            },
        error: function(data) {
            alert("error")
            console.log(data)
            }
        })
})
    
var expandFen = function(fen) {
    var move = document.querySelector('input[name="move"]:checked').value;
    var castle = "-"
    var ep = "-"
    var halfmove = "0"
    var fullmove = "1"
    var sep = " "
    var toAdd = [move, castle, ep, halfmove, fullmove]

    for (var i=0; i < toAdd.length; i++) {
        fen += sep
        fen += toAdd[i]
    }

    return fen
}

var setFEN = function(fen) {
    
    orientation = document.getElementById("reversed-input").checked ? "black" : "white"
    board.orientation(orientation)
    board.position(fen, true)
    
}


$(document).ready(init);