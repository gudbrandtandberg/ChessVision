// 
// main.js -- all the chessvision website stuff
//

// global variablaes
var board, cropper, cropperOptions;
var input, canvas, context, endpoint;
var endpoint, cv_algo_url, analyze_url;

var sizeCanvas = function() {
    container = document.getElementById("preview-container");
    container.style.height = container.offsetWidth
}

// initialize variables
var init = function() {

    sizeCanvas()
    $("#analyze-btn").on("click", requestAnalysis)
    // $("#preview-anchor").on("click", alert("click"))

    endpoint = document.getElementById("endpoint").innerHTML
    cv_algo_url = endpoint + "cv_algo/"
    analyze_url = endpoint + "analyze/"
    
    // Initialize chessboard (chessboard.js)
    board = ChessBoard("board", {position: "start",
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
        $("#flip-pane").show()
        $("#to-move").show()
        $("#needle-wrapper").hide()
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
    setSpinner()

    // option: minWidth
    if (cropper == undefined) {
        alert("Please upload a photo first!");
        return;
    }
    dataURL = cropper.getCroppedCanvas({width: 512, height: 512}).toDataURL('image/jpeg', 0.9)

    var blobBin = atob(dataURL.split(',')[1]);
    var array = [];
    for (var i = 0; i < blobBin.length; i++) {
        array.push(blobBin.charCodeAt(i));
    }
    var file = new Blob([new Uint8Array(array)], {type: 'image/jpeg'});
    flip = document.getElementById("reversed-input").checked ? "true" : "false"
    var tomove = document.querySelector('input[name="move"]:checked').value;
    var formData = new FormData();
    formData.append("file", file);
    formData.append("flip", flip);
    formData.append("tomove", tomove);

    $.ajax({
        url: cv_algo_url,
        method: "POST", 
        data: formData,
        cache: false,
        contentType: false,
        processData: false,
        timeout: 50000,
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
        $("#preview-container").hide()
        $("#edit-analyze-pane").show()
        board.resize()
        setFEN(res.FEN)
        document.getElementById("raw-id-input").value = res.id
        $("#needle-wrapper").show()
        if (res.score != "None") {
            setScore(res.score)
        } else if (res.mate != "None") {
            setMate(res.mate)
        } else {
            alert("Position is invalid, cannot provide analysis")
        }
        $("#flip-pane").hide()
        
    } else {
        alert(res.errorMsg)
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

var requestAnalysis = function() {
    
    //var formData = new FormData();
    // get valid fen from board + input tags.
    var fen = board.fen()
    fen = expandFen(fen)
    var formData = {"FEN": fen};
    
    $.ajax({
        url: analyze_url,
        method: "POST",
        data: formData,
        success: function(data) {
            res = JSON.parse(data)
            if (res.success == "false") {
                alert("Analysis failed..")
                return
            }
    
            var bestMove = res.bestMove
            var score, mate
    
            if (bestMove == "(none)") {
                return
            }

            if (res.score != "None") {
                score = parseFloat(res.score)
                setScore(score)
                
            } else {
                mate = parseInt(res.mate)
                setMate(mate)
            }
            
            if (bestMove.length == 5) {
                alert("strange move!")
            }
            src = bestMove.substr(0, 2)
            dst = bestMove.substr(2, 2)
            move = src.concat("-", dst)
            console.log("Best move is: " + move)
            board.move(move)
            toggleTurn()

            },
        error: function(data) {
            alert("error")
            console.log(data)
            }
        })
}

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

var setMate = function(mate) {
    var tomove = document.querySelector('input[name="move"]:checked').value;
    
    if ((mate > 0 && tomove == "w") || (mate < 0 && tomove == "b")) {
        $("#needle-content").css({width: "100%"})
    } else {
        $("#needle-content").css({width: "0%"})
    }
    absMate = Math.abs(mate)
    displayScore = "#" + absMate.toString()
    $("#needle-content").attr("aria-valuenow", displayScore)
    $("#needle-content").html(displayScore)
}

var setScore = function(score) {

    displayScore = parseFloat(score)
    if (score > 4.0) {
        displayScore = 4.0
    } else if (score < -4.0) {
        displayScore = -4.0
    }
    displayScore += 4.0
    width = (displayScore * 100.0 / 8.0).toString() + "%"

    $("#needle-content").html(score.toString())
    $("#needle-content").attr("aria-valuenow", displayScore)
    $("#needle-content").css({width: width})
}

$(document).ready(init);