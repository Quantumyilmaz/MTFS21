anime.timeline({loop: false})
  .add({
    targets: '.ml5 .line',
    opacity: [0.5,1],
    scaleX: [0, 1],
    easing: "easeInOutExpo",
    duration: 300
  }).add({
    targets: '.ml5 .line',
    duration: 300,
    easing: "easeOutExpo",
    translateY: (el, i) => (-0.625 + 0.625*2*i) + "em"
  }).add({
    targets: '.ml5 .ampersand',
    opacity: [0,1],
    scaleY: [0.5, 1],
    easing: "easeInExpo",
    duration: 200,
    offset: '-=0'
  }).add({
    targets: '.ml5 .letters-left',
    opacity: [0,1],
    translateX: ["0.5em", 0],
    easing: "easeOutExpo",
    duration: 100,
    offset: '-=0'
  }).add({
    targets: '.ml5 .letters-right',
    opacity: [0,1],
    translateX: ["-0.5em", 0],
    easing: "easeOutExpo",
    duration: 100,
    offset: '-=0'
  }).add({
    targets: '.ml5',
    opacity: 1,
    duration: 1000,
    easing: "easeOutExpo",
    delay: 1000
  });

  // Wrap every letter in a span
var textWrapper = document.querySelector('.ml12');
textWrapper.innerHTML = textWrapper.textContent.replace(/\S/g, "<span class='letter'>$&</span>");

anime.timeline({loop: false})
  .add({
    targets: '.ml12 .letter',
    translateX: [40,0],
    translateZ: 0,
    opacity: [0,1],
    easing: "easeOutExpo",
    duration: 1200,
    delay: (el, i) => 500 + 30 * i
  });

anime.timeline({loop: false})
  .add({
    targets: '.column',
    // translateX: [40,0],
    // translateZ: [40,1000],
    opacity: [0,1],
    easing: "easeOutExpo",
    duration: 1200,
    delay: (el, i) => 1000 + 30 * i
  });

  
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

window.mobileCheck = function() {
    let check = false;
    (function(a){if(/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(a)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(a.substr(0,4))) check = true;})(navigator.userAgent||navigator.vendor||window.opera);
    return check;
  };

// document.getElementById("demo").innerHTML = window.mobileCheck();
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
        if (window.mobileCheck()) {
            myFunc(id+"2")
        }
    }

    // main.style.width = (screen.width-sidenav.offsetWidth).toString() + "px";
    // main.style.height = (screen.height - topnav.offsetHeight).toString() + "px";
    // main.style.marginLeft = (sidenav.offsetWidth-8).toString() + "px";
    // main.style.marginTop = (topnav.offsetHeight).toString() + "px";
    // if (id==="litsum"){
    //     myFunc(id+"2");
    // }
}

//###########   MAIN   ##############
var i;
var sidelinks = document.getElementById("sidenav").getElementsByTagName("a");
for (i=0;i<sidelinks.length;i++) {
    if (sidelinks[i].hasAttribute("id") && window.location.href.includes(sidelinks[i].id)){
        myFunc(sidelinks[i].id);
        break;
    }
}

var dropdown = document.getElementsByClassName("dropdown-btn");
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


