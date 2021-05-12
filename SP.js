
/* Loop through all dropdown buttons to toggle between hiding and showing its dropdown content - This allows the user to have multiple dropdowns without any conflict */
var dropdown = document.getElementsByClassName("dropdown-btn");
var sidenav = document.getElementById("sidenav");
var topnav = document.getElementById("topnav");
var mains = document.getElementsByClassName("main");
var main  = document.getElementById("main");
var i;
var parmak_payi = 40;
main.style.marginLeft = (sidenav.offsetWidth+parmak_payi).toString() + "px";
topnav.style.marginLeft = sidenav.offsetWidth.toString() + "px";
topnav.style.width = (screen.width - sidenav.offsetWidth).toString() + "px";


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

document.getElementById("demo").innerHTML= main.style.padding;


function myFunc(id){

    main.style.display = "none";
    main = document.getElementById("main_"+id);
    main.style.display="block";

    main.style.width = (screen.width-sidenav.offsetWidth).toString() + "px";
    main.style.height = (screen.height - topnav.offsetHeight).toString() + "px";
    main.style.marginLeft = (sidenav.offsetWidth-8).toString() + "px";
    main.style.marginTop = (topnav.offsetHeight).toString() + "px";
}

