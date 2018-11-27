function res = tonemapLAB_simple(lab,L0,L,val0,val2,exposure,gamma,saturation)
if val0==0
    diff0 = L-L0;
else
    if val0>0
        diff0 = sigmoid((L-L0)/100,val0)*100;
    else
        diff0 = (1+val0)*(L-L0);
    end
end

if val2==0
    base = exposure*L0;
else
    if val2>0
        base = (sigmoid((exposure*L0-56)/100,val2)*100)+56;
    else
        base = (1+val2)*(exposure*L0-56) + 56;
    end
end

if gamma == 1
    res = base + diff0;
else
    maxBase = max(base(:));
    res = (zeroone(base).^gamma)*maxBase + diff0;
end

if saturation == 0
    lab(:,:,1) = res;
else
    lab(:,:,1) = res;
    lab(:,:,2) = lab(:,:,2) * saturation;
    lab(:,:,3) = lab(:,:,3) * saturation;
end

cform = makecform('lab2srgb');
res = applycform(lab, cform);