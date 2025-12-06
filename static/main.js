$(document).ready(function () {

    // ===============================
    // Mobile Menu Toggle
    // ===============================
    $('#menu-bar').click(function () {
        $(this).toggleClass('fa-times');
        $('.navbar').toggleClass('nav-toggle');
    });

    // ===============================
    // Navbar Active Link on Scroll
    // ===============================
    $(window).on('load scroll', function () {

        $('#menu-bar').removeClass('fa-times');
        $('.navbar').removeClass('nav-toggle');

        $('section').each(function () {

            let top = $(window).scrollTop();
            let height = $(this).height();
            let id = $(this).attr('id');
            let offset = $(this).offset().top - 200;

            if (top > offset && top < offset + height) {
                $('.navbar ul li a').removeClass('active');
                $('.navbar').find('[href="#' + id + '"]').addClass('active');
            }
        });
    });

    // ===============================
    // (Optional) Image switching logic
    // If you don't use .list .btn, this will simply do nothing.
    // ===============================
    $('.list .btn').click(function () {
        $(this).addClass('active').siblings().removeClass('active');
        let src = $(this).attr('data-src');
        $('.menu .row .image img').attr('src', src);
    });

    // ===============================
    // Predict button "loading" state
    // ===============================
    $('form').on('submit', function () {
        const $btn = $(this).find('.my-cta-button');

        if ($btn.length) {
            const originalText = $btn.val() || $btn.text();
            $btn.data('original-text', originalText);

            if ($btn.is('input')) {
                $btn.val('Predicting...');
            } else {
                $btn.text('Predicting...');
            }

            $btn.prop('disabled', true).addClass('loading');
        }
    });
});
