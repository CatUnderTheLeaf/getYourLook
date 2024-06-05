var imageUrls = Array.prototype.map.call(document.images, function (i) {
    if ((i.src.split('/')).slice(-1)[0].split('.')[1]=='jpg') {
        return i.src;
    }
});
console.log(imageUrls.join('\r\n'));