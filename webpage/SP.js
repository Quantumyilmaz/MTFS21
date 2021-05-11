/* Loop through all dropdown buttons to toggle between hiding and showing its dropdown content - This allows the user to have multiple dropdowns without any conflict */
var dropdown = document.getElementsByClassName("dropdown-btn");
var sidenav = document.getElementById("sidenav");
var topnav = document.getElementById("topnav");
var main  = document.getElementById("main")
var i;

for (i = 0; i < dropdown.length; i++) {
    dropdown[i].addEventListener("click", function() {
    this.classList.toggle("active");
    var dropdownContent = this.nextElementSibling;
    if (dropdownContent.style.display === "block") {
    dropdownContent.style.display = "none";
    topnav.style.marginLeft = sidenav.offsetWidth.toString() + "px";
    main.style.marginLeft = sidenav.offsetWidth.toString() + "px";
    } else {
    dropdownContent.style.display = "block";
    topnav.style.marginLeft = dropdownContent.offsetWidth.toString() + "px";
    main.style.marginLeft = dropdownContent.offsetWidth.toString() + "px";
    }
    });
}

var parmak_payi = 40 
topnav.style.marginLeft = sidenav.offsetWidth.toString() + "px";
topnav.style.width = (screen.width - sidenav.offsetWidth).toString() + "px";
topnav.style.right = (main.innerWidth + parmak_payi).toString() + "px";
main.style.marginLeft = (sidenav.offsetWidth+parmak_payi).toString() + "px";
topnav.style.width = (screen.width - sidenav.offsetWidth).toString() + "px";