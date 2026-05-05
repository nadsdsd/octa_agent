class unionfind:
    def __int__(self,n:int):
        self.parent =list(range(n))
    def find(self,x:int):
        if x!=self.parent[x]:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self,x,y):
        p_x = self.find(x)
        p_y  =self.find(y)
        if p_x==p_y:
            return False
        else:
            self.parent[p_x]=  p_y
            return True
        



