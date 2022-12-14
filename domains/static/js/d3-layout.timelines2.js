(function() {
//d3.layout.timeline = function() {
  d3.timeline = function() {
    var timelines = [];
    var dateAccessor = function (d) {return new Date(d)};
    var processedTimelines = [];
    var startAccessor = function (d) {return d.startDate};
    var endAccessor = function (d) {return d.endDate};
    var size = [500, 300];
    var timelineExtent = [-Infinity, Infinity];
    var setExtent = [];
    var displayScale = d3.scaleLinear();
    var swimlanes = [];
    var padding = 0;
    var fixedExtent = false;
    var maximumHeight = Infinity;

    function processTimelines() {
        timelines.forEach(function (band) {
            var projectedBand = {};
            for (var x in band) {
                if (band.hasOwnProperty(x)) {
                    projectedBand[x] = band[x];
                }
            }
            projectedBand.startDate = dateAccessor(startAccessor(band));
            projectedBand.endDate = dateAccessor(endAccessor(band));
            projectedBand.lane = 0;
            processedTimelines.push(projectedBand);
        });
    }

    function projectTimelines() {
        if (fixedExtent === false) {
            var minStart = d3.min(processedTimelines, function (d) {return d.startDate});
            var maxEnd = d3.max(processedTimelines, function (d) {return d.endDate});
            timelineExtent = [minStart,maxEnd];
        }
        else {
            timelineExtent = [dateAccessor(setExtent[0]), dateAccessor(setExtent[1])];
        }

        displayScale.domain(timelineExtent).range([0,size[0]]);

        processedTimelines.forEach(function (band) {
            band.originalStart = band.startDate;
            band.originalEnd = band.endDate;
            band.startDate = displayScale(band.startDate);
            band.endDate = displayScale(band.endDate);
        });
    }

    function fitsIn(lane, band) {
        if (lane.endDate < band.startDate || lane.startDate > band.endDate) {
            return true;
        }
        var filteredLane = lane.filter(function (d) {return d.startDate <= band.endDate && d.endDate >= band.startDate});
        if (filteredLane.length === 0) {
            return true;
        }
        return false;
    }

    function findlane(band) {
        //make the first array
        if (swimlanes[0] === undefined) {
            swimlanes[0] = [band];
            return;
        }
        var l = swimlanes.length - 1;
        var x = 0;

        while (x <= l) {
            if (fitsIn(swimlanes[x], band)) {
                swimlanes[x].push(band);
                return;
            }
            x++;
        }
        swimlanes[x] = [band];
        return;
    }

    function timeline(data) {
        if (!arguments.length) return timeline;

        timelines = data;

        processedTimelines = [];
        swimlanes = [];

        processTimelines();
        projectTimelines();


        processedTimelines.forEach(function (band) {
            findlane(band);
        });

        var height = size[1] / swimlanes.length;
        height = Math.min(height, maximumHeight);

        swimlanes.forEach(function (lane, i) {
            lane.forEach(function (band) {
                band.y = i * (height);
                band.dy = height - padding;
                band.lane = i;
            });
        });

        return processedTimelines;
    }

    timeline.dateFormat = function (_x) {
         if (!arguments.length) return dateAccessor;
         dateAccessor = _x;
        return timeline;
    }

    timeline.bandStart = function (_x) {
         if (!arguments.length) return startAccessor;
         startAccessor = _x;
        return timeline;
    }

    timeline.bandEnd = function (_x) {
         if (!arguments.length) return endAccessor;
         endAccessor = _x;
        return timeline;
    }

    timeline.size = function (_x) {
         if (!arguments.length) return size;
         size = _x;
        return timeline;
    }

    timeline.padding = function (_x) {
         if (!arguments.length) return padding;
         padding = _x;
        return timeline;
    }

    timeline.extent = function (_x) {
        if (!arguments.length) return timelineExtent;
            fixedExtent = true;
            setExtent = _x;
            if (_x.length === 0) {
                fixedExtent = false;
            }
        return timeline;
    }

    timeline.maxBandHeight = function (_x) {
        if (!arguments.length) return maximumHeight;
            maximumHeight = _x;
        return timeline;
    }

    return timeline;
}
})();