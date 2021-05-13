function resize()
{
    var sidenav = document.getElementById("sidenav");
    // var topnav = document.getElementById("topnav");
    // var mains = document.getElementsByClassName("main");
    // var main  = document.getElementById("main");
    var i;
    var j;
    // main.style.width = (screen.width-sidenav.offsetWidth).toString() + "px";
    // main.style.height = (screen.height - topnav.offsetHeight).toString() + "px";
    // main.style.marginLeft = (sidenav.offsetWidth).toString() + "px";
    // main.style.marginTop = (topnav.offsetHeight+100).toString() + "px";
    // topnav.style.marginLeft = sidenav.offsetWidth.toString() + "px";
    // topnav.style.width = 100*(1 - sidenav.offsetWidth/screen.width).toString() + "%";

    // document.getElementById("demo").innerHTML = document.getElementsByClassName("column")[1].getElementsByClassName("column_mini").length;
    var columns = document.getElementsByClassName("column");
    for (i = 0; i<columns.length; i++) {
        columns[i].style.width = (100/columns.length)*(1 - sidenav.offsetWidth/screen.width).toString() + "%";
        columns[i].style.marginLeft = (100/(columns.length+1))*(sidenav.offsetWidth/screen.width).toString() + "%";
        var childrenz = columns[i].getElementsByClassName("column_mini");
        for (j=0;j<childrenz.length;j++){
            childrenz[j].style.width = (100/childrenz.length).toString() + "%";
        }
    }

}

// (function() {
//     window.onresize = displayWindowSize;
//     window.onload = displayWindowSize;
  
//     function displayWindowSize() {
//       let myWidth = window.innerWidth;
//       let myHeight = window.innerHeight;
//       // your size calculation code here
//       document.getElementById("demo").innerHTML = myWidth + "x" + myHeight;
//     };
  
  
//   })();

var new_main;

function myFunc(id){
    
    if (new_main){
        new_main.style.display = "none";
    } else {
        main.style.display = "none";
    }
    new_main = document.getElementById("main_"+id);
    new_main.style.display="block";
    if (new_main.className==="pdf"){
        new_main.style.height = (screen.height - topnav.offsetHeight).toString() + "px";
    }

    // main.style.width = (screen.width-sidenav.offsetWidth).toString() + "px";
    // main.style.height = (screen.height - topnav.offsetHeight).toString() + "px";
    // main.style.marginLeft = (sidenav.offsetWidth-8).toString() + "px";
    // main.style.marginTop = (topnav.offsetHeight).toString() + "px";
    // if (id==="litsum"){
    //     myFunc(id+"2");
    // }
}

var dropdown = document.getElementsByClassName("dropdown-btn");
var i;
for (i = 0; i < dropdown.length; i++) {
    dropdown[i].addEventListener("click", function() {
    this.classList.toggle("active");
    var dropdownContent = this.nextElementSibling;
    if (dropdownContent.style.display === "block") {
    dropdownContent.style.display = "none";
    // topnav.style.marginLeft = sidenav.offsetWidth.toString() + "px";
    // main.style.marginLeft = sidenav.offsetWidth.toString() + "px";
    } else {
    dropdownContent.style.display = "block";
    };
    });
};
resize()
