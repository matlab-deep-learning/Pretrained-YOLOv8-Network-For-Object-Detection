function boxesXYWH = dist2bbox(distance,anchorGrid)
% Transform distance(ltrb) to box(xywh or xyxy)

xywh = true;
dim=1;

% matrix divided into two along dim 1
rb = distance(3:end,:);
lt = distance(1:2,:);
x1y1 = anchorGrid - lt;
x2y2 = anchorGrid + rb;

if xywh
    cxy = (x1y1 + x2y2) ./ 2;
    wh = x2y2 - x1y1;
end

boxesXYWH = vertcat(cxy, wh);  % xywh bbox
end