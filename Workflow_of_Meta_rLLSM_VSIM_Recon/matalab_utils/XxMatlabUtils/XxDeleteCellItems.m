function [cell_removed, I] = XxDeleteCellItems(cell_raw, remove_filter)

nitems = max(size(cell_raw));
flag = ones(size(cell_raw));
for i = 1:nitems
    cur_item = cell_raw{i};
    if contains(cur_item,remove_filter)
        flag(i) = 0;
    end
end

index = 1:nitems;
cell_removed = cell_raw(logical(flag));
I = index(logical(flag));