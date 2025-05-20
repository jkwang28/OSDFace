window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 5000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();

})

// 添加返回顶部按钮的功能
document.addEventListener('DOMContentLoaded', function() {
  // 获取返回顶部按钮
  const backToTopButton = document.getElementById('back-to-top');
  
  // 监听滚动事件
  window.addEventListener('scroll', function() {
    if (window.pageYOffset > 300) { // 当页面滚动超过300px时显示按钮
      backToTopButton.style.display = 'block';
    } else {
      backToTopButton.style.display = 'none';
    }
  });
  
  // 点击按钮返回顶部
  backToTopButton.addEventListener('click', function() {
    window.scrollTo({
      top: 0,
      behavior: 'smooth' // 平滑滚动
    });
  });
});

// 图片加载优化
document.addEventListener('DOMContentLoaded', function() {
  const images = document.querySelectorAll('img');
  images.forEach(img => {
    img.addEventListener('load', function() {
      this.style.opacity = 1;
    });
  });
});

// 图片懒加载功能
document.addEventListener('DOMContentLoaded', function() {
  const lazyImages = document.querySelectorAll('img[data-src]');
  
  const imageObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const img = entry.target;
        img.src = img.dataset.src; // 将 data-src 的值赋给 src
        img.classList.add('loaded');
        observer.unobserve(img); // 加载完成后取消观察
      }
    });
  }, {
    rootMargin: '50px 0px' // 提前50px开始加载
  });

  lazyImages.forEach(img => {
    imageObserver.observe(img);
  });
});
