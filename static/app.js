$(document).ready(function() {
    $("#getStartedBtn").click(function() {
        $('html, body').animate({
            scrollTop: $("#watermark").offset().top
        }, 1000);
    });

    $("#watermarkForm").submit(function(e) {
        e.preventDefault();
        var formData = new FormData(this);
        $.ajax({
            url: $(this).attr('action'),
            type: $(this).attr('method'),
            data: formData,
            processData: false,
            contentType: false,
            success: function(data) {
                console.log(data);
                // Handle success here
            },
            error: function(error) {
                console.log(error);
                // Handle error here
            }
        });
    });

    $("#detectForm").submit(function(e) {
        e.preventDefault();
        var formData = new FormData(this);
        $.ajax({
            url: $(this).attr('action'),
            type: $(this).attr('method'),
            data: formData,
            processData: false,
            contentType: false,
            success: function(data) {
                console.log(data);
                // Handle success here
            },
            error: function(error) {
                console.log(error);
                // Handle error here
            }
        });
    });
});
