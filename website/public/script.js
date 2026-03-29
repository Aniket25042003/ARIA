/* ─── ARIA Landing Page — Script ─── */

(function () {
  "use strict";

  // ─── Intersection Observer: scroll animations ───
  const scrollElements = document.querySelectorAll(".animate-on-scroll");

  const scrollObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          scrollObserver.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12, rootMargin: "0px 0px -40px 0px" }
  );

  scrollElements.forEach((el) => scrollObserver.observe(el));

  // ─── Navbar scroll behaviour ───
  const navbar = document.getElementById("navbar");
  let lastScroll = 0;

  function onScroll() {
    const y = window.scrollY;
    if (y > 50) {
      navbar.classList.add("scrolled");
    } else {
      navbar.classList.remove("scrolled");
    }
    lastScroll = y;
  }

  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  // ─── Mobile nav toggle ───
  const navToggle = document.getElementById("navToggle");
  const navLinks = document.querySelector(".nav-links");

  if (navToggle && navLinks) {
    navToggle.addEventListener("click", () => {
      navToggle.classList.toggle("active");
      navLinks.classList.toggle("open");
    });

    // Close mobile menu when a link is clicked
    navLinks.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", () => {
        navToggle.classList.remove("active");
        navLinks.classList.remove("open");
      });
    });
  }

  // ─── Hero stat counter animation ───
  const statValues = document.querySelectorAll(".hero-stat-value[data-count]");

  function animateCounter(el) {
    const target = parseInt(el.getAttribute("data-count"), 10);
    const duration = 1600;
    const start = performance.now();

    function tick(now) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      // Ease-out cubic
      const ease = 1 - Math.pow(1 - progress, 3);
      el.textContent = Math.round(target * ease);
      if (progress < 1) requestAnimationFrame(tick);
    }

    requestAnimationFrame(tick);
  }

  const statObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          animateCounter(entry.target);
          statObserver.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.5 }
  );

  statValues.forEach((el) => statObserver.observe(el));

  // ─── Smooth scroll for anchor links ───
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", (e) => {
      const href = anchor.getAttribute("href");
      if (href === "#") return;
      const target = document.querySelector(href);
      if (target) {
        e.preventDefault();
        const navHeight = navbar.offsetHeight;
        const top =
          target.getBoundingClientRect().top + window.scrollY - navHeight - 16;
        window.scrollTo({ top, behavior: "smooth" });
      }
    });
  });

  // ─── Waitlist form handler ───
  window.handleWaitlist = function (event) {
    event.preventDefault();
    const form = event.target;
    const input = form.querySelector(".email-input");
    const hint = document.getElementById("emailHint");
    const email = input.value.trim();

    if (!email) return;

    // Disable form
    input.disabled = true;
    form.querySelector("button").disabled = true;

    // Simulate submission (replace with real endpoint later)
    setTimeout(() => {
      input.value = "";
      input.disabled = false;
      form.querySelector("button").disabled = false;
      hint.textContent = "You're on the list! We'll let you know when ARIA launches.";
      hint.style.color = "var(--accent)";
      hint.style.fontWeight = "600";
    }, 800);
  };

  // ─── Parallax effect on hero orbs ───
  const hero = document.getElementById("hero");
  const orbs = hero ? hero.querySelectorAll(".hero-orb") : [];

  if (orbs.length && window.matchMedia("(min-width: 768px)").matches) {
    window.addEventListener(
      "mousemove",
      (e) => {
        const x = (e.clientX / window.innerWidth - 0.5) * 2;
        const y = (e.clientY / window.innerHeight - 0.5) * 2;
        orbs.forEach((orb, i) => {
          const speed = (i + 1) * 12;
          orb.style.transform = `translate(${x * speed}px, ${y * speed}px)`;
        });
      },
      { passive: true }
    );
  }

  // ─── Tech card tilt effect on hover ───
  document.querySelectorAll(".tech-card").forEach((card) => {
    card.addEventListener("mousemove", (e) => {
      const rect = card.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width - 0.5) * 8;
      const y = ((e.clientY - rect.top) / rect.height - 0.5) * -8;
      card.style.transform = `perspective(600px) rotateY(${x}deg) rotateX(${y}deg) translateY(-4px)`;
    });
    card.addEventListener("mouseleave", () => {
      card.style.transform = "";
    });
  });

  // ─── Feature card glow follow cursor ───
  document.querySelectorAll(".feature-card").forEach((card) => {
    const glow = card.querySelector(".feature-card-glow");
    if (!glow) return;
    card.addEventListener("mousemove", (e) => {
      const rect = card.getBoundingClientRect();
      glow.style.opacity = "1";
      glow.style.left = e.clientX - rect.left + "px";
      glow.style.top = e.clientY - rect.top + "px";
    });
    card.addEventListener("mouseleave", () => {
      glow.style.opacity = "0";
    });
  });
})();
